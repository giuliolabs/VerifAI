# AWS EC2 Setup Guide (Docker + Docker Compose)

This guide explains how to deploy a Dockerized full-stack application (backend + frontend)
on an AWS EC2 instance using Docker and Docker Compose.

---

## 1. Create an EC2 Instance

1. Log in to the AWS Console
2. Go to **EC2 → Launch instance**
3. Configure the instance:
   - **Name**: your-project-name
   - **AMI**: Ubuntu Server 22.04 LTS
   - **Instance type**: t2.micro (Free Tier eligible)
   - **Key pair**: create or select an existing one
   - **Storage**: default is fine (8–16 GB)

### Security Group (IMPORTANT)
Allow the following inbound rules:

|   Type   |   Protocol   |   Port   |   Source   |
|----------|--------------|----------|------------|
|   SSH    |     TCP      |    22    |  Your IP   |
|   HTTP   |     TCP      |    80    |  0.0.0.0/0 |
| Frontend |     TCP      |   3000   |  0.0.0.0/0 |
| Backend  |     TCP      |   8000   |  0.0.0.0/0 |

Launch the instance.

---

## 2. Connect to the Instance

From your local machine:

```bash
ssh -i your-key.pem ubuntu@EC2_PUBLIC_IP
