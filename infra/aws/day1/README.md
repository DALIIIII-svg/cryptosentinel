# Day-1 AWS checks

Run these in your **first Learner Lab session**, before building anything on AWS.
They answer the questions that decide the design. Write each result in the root
`README.md` under **Environment facts**.

| Check | Question | Where to run | Command |
| --- | --- | --- | --- |
| A | Can us-east-1 reach Binance market data? | a throwaway EC2 `t3.micro` (Amazon Linux 2023, `LabInstanceProfile`) | `bash check_binance.sh` |
| B | Is Bedrock usable? | your laptop (WSL) | `bash infra/aws/day1/check_bedrock.sh` |
| C | Can we train a built-in algorithm and serve it **serverless**? | your laptop (WSL) | `.venv/bin/python infra/aws/day1/check_sagemaker.py` |
| D | Is Kafka on EC2 accepted instead of MSK? Is a free external LLM API (Groq) accepted if Bedrock is blocked? | message to the teacher | — |

## Before checks B and C: give your laptop the Learner Lab credentials

1. Learner Lab → **Start Lab** → wait for the green dot.
2. **AWS Details** → **AWS CLI: Show** → copy the three lines.
3. In WSL: `mkdir -p ~/.aws && nano ~/.aws/credentials`, paste under a `[default]` line, save.
4. `printf "[default]\nregion = us-east-1\n" > ~/.aws/config`
5. Test: `aws sts get-caller-identity`

These credentials expire when the session ends: repeat steps 2–3 every session.

## Check A on EC2, step by step

1. EC2 → Launch instance → name `cs-day1`, Amazon Linux 2023, `t3.micro`, no key pair needed.
2. Advanced details → IAM instance profile → `LabInstanceProfile`.
3. Launch, then select it → **Connect** → **EC2 Instance Connect** → Connect.
4. Paste the content of `check_binance.sh` into a file (`nano check_binance.sh`), then `bash check_binance.sh`.
5. **Terminate** the instance when done.

## Cost

Check C costs a few cents (one ~4-minute `ml.m5.large` training job and a short-lived endpoint).
The script deletes everything it created, even if it fails. If it prints
"could not delete", delete that resource in the SageMaker console.
