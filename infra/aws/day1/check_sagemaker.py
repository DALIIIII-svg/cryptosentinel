"""Day-1 Check C: can this Learner Lab train with a SageMaker built-in algorithm
and serve it from a SERVERLESS endpoint?

What it does (about 10 minutes, a few cents):
  1. creates a temporary bucket and uploads 200 rows of toy data
  2. trains built-in XGBoost 1.7 on ml.m5.large (training job)
  3. deploys it to a serverless endpoint (or ml.t2.medium with --realtime)
  4. sends one prediction
  5. DELETES everything it created (endpoint, config, model, bucket), even on failure

Run on your laptop (WSL) after copying the Learner Lab credentials:
    .venv/bin/python infra/aws/day1/check_sagemaker.py
    .venv/bin/python infra/aws/day1/check_sagemaker.py --realtime   # if serverless is refused
"""
from __future__ import annotations

import argparse
import random
import sys
import time

import boto3
from botocore.exceptions import ClientError

# Built-in XGBoost 1.7-1 image accounts (AWS docs, "Docker registry paths").
XGBOOST_IMAGE = {
    "us-east-1": "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:1.7-1",
    "us-west-2": "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-xgboost:1.7-1",
}
TAGS = [{"Key": "project", "Value": "cryptosentinel"}, {"Key": "purpose", "Value": "day1-check"}]


def toy_csv(n: int = 200) -> str:
    rnd = random.Random(42)
    rows = []
    for _ in range(n):
        x = rnd.uniform(0, 10)
        rows.append(f"{2 * x + rnd.gauss(0, 0.5):.4f},{x:.4f}")  # label first, no header
    return "\n".join(rows) + "\n"


def wait_training(sm, name: str) -> str:
    while True:
        desc = sm.describe_training_job(TrainingJobName=name)
        status = desc["TrainingJobStatus"]
        print(f"  training: {status} / {desc.get('SecondaryStatus')}")
        if status in ("Completed", "Failed", "Stopped"):
            if status != "Completed":
                raise RuntimeError(desc.get("FailureReason", status))
            return desc["ModelArtifacts"]["S3ModelArtifacts"]
        time.sleep(30)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--realtime", action="store_true", help="use an ml.t2.medium endpoint instead of serverless")
    args = ap.parse_args()

    session = boto3.Session(region_name=args.region)
    account = session.client("sts").get_caller_identity()["Account"]
    role_arn = f"arn:aws:iam::{account}:role/LabRole"
    s3 = session.client("s3")
    sm = session.client("sagemaker")
    smr = session.client("sagemaker-runtime")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    bucket = f"cryptosentinel-day1-{account}"
    job = f"cs-day1-xgb-{stamp}"
    model = endpoint = config = job
    created = {"bucket": False, "model": False, "config": False, "endpoint": False}
    result = 1

    try:
        print(f"Account {account}, role {role_arn}, region {args.region}")
        print("1. Bucket + toy data")
        if args.region == "us-east-1":
            s3.create_bucket(Bucket=bucket)
        else:
            s3.create_bucket(Bucket=bucket, CreateBucketConfiguration={"LocationConstraint": args.region})
        created["bucket"] = True
        s3.put_object(Bucket=bucket, Key="day1/train/train.csv", Body=toy_csv().encode())

        print("2. Training job (built-in XGBoost, ml.m5.large) - about 4 minutes")
        sm.create_training_job(
            TrainingJobName=job,
            RoleArn=role_arn,
            AlgorithmSpecification={"TrainingImage": XGBOOST_IMAGE[args.region], "TrainingInputMode": "File"},
            HyperParameters={"objective": "reg:squarederror", "num_round": "20"},
            InputDataConfig=[{
                "ChannelName": "train",
                "ContentType": "text/csv",
                "DataSource": {"S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": f"s3://{bucket}/day1/train/",
                    "S3DataDistributionType": "FullyReplicated",
                }},
            }],
            OutputDataConfig={"S3OutputPath": f"s3://{bucket}/day1/output/"},
            ResourceConfig={"InstanceType": "ml.m5.large", "InstanceCount": 1, "VolumeSizeInGB": 5},
            StoppingCondition={"MaxRuntimeInSeconds": 900},
            Tags=TAGS,
        )
        artifact = wait_training(sm, job)
        print("  [PASS] training works")

        print("3. Model + endpoint", "(real-time ml.t2.medium)" if args.realtime else "(serverless)")
        sm.create_model(
            ModelName=model,
            ExecutionRoleArn=role_arn,
            PrimaryContainer={"Image": XGBOOST_IMAGE[args.region], "ModelDataUrl": artifact},
            Tags=TAGS,
        )
        created["model"] = True
        variant = {"VariantName": "main", "ModelName": model}
        if args.realtime:
            variant.update(InstanceType="ml.t2.medium", InitialInstanceCount=1)
        else:
            variant["ServerlessConfig"] = {"MemorySizeInMB": 2048, "MaxConcurrency": 1}
        sm.create_endpoint_config(EndpointConfigName=config, ProductionVariants=[variant], Tags=TAGS)
        created["config"] = True
        sm.create_endpoint(EndpointName=endpoint, EndpointConfigName=config, Tags=TAGS)
        created["endpoint"] = True
        print("  waiting for the endpoint (3 to 8 minutes)...")
        sm.get_waiter("endpoint_in_service").wait(
            EndpointName=endpoint, WaiterConfig={"Delay": 30, "MaxAttempts": 30}
        )

        print("4. Prediction for x = 3.0 (expected about 6)")
        t0 = time.time()
        resp = smr.invoke_endpoint(EndpointName=endpoint, ContentType="text/csv", Body=b"3.0")
        pred = resp["Body"].read().decode().strip()
        print(f"  [PASS] prediction {pred} in {time.time() - t0:.1f} s (first call includes cold start)")
        result = 0
    except ClientError as exc:
        print(f"  [FAIL] {exc.response['Error'].get('Code')}: {exc.response['Error'].get('Message')}")
        if not args.realtime and created["model"]:
            print("  => If serverless was refused, run again with --realtime and note it in README.md")
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] {type(exc).__name__}: {exc}")
    finally:
        print("5. Cleanup")
        for label, fn in (
            ("endpoint", lambda: sm.delete_endpoint(EndpointName=endpoint)),
            ("config", lambda: sm.delete_endpoint_config(EndpointConfigName=config)),
            ("model", lambda: sm.delete_model(ModelName=model)),
        ):
            if created[label]:
                try:
                    fn()
                    print(f"  deleted {label}")
                except ClientError as exc:
                    print(f"  could not delete {label}: {exc} -> delete it in the console!")
        if created["bucket"]:
            try:
                for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket):
                    for obj in page.get("Contents", []):
                        s3.delete_object(Bucket=bucket, Key=obj["Key"])
                s3.delete_bucket(Bucket=bucket)
                print("  deleted bucket")
            except ClientError as exc:
                print(f"  could not delete bucket {bucket}: {exc}")
    print("RESULT:", "PASS" if result == 0 else "FAIL")
    return result


if __name__ == "__main__":
    sys.exit(main())
