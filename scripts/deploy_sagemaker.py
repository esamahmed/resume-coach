"""
scripts/deploy_sagemaker.py

Deploy Llama-2 or Mistral-7B on AWS SageMaker (ml.g5.2xlarge).
This is the LLM hosting layer for the Resume Coach capstone.

Usage:
    python scripts/deploy_sagemaker.py --model llama2     # Deploy Llama-2-7B-Chat
    python scripts/deploy_sagemaker.py --model mistral    # Deploy Mistral-7B-Instruct
    python scripts/deploy_sagemaker.py --delete           # Delete endpoint (save costs!)

Cost reminder:
    ml.g5.2xlarge = ~$1.52/hr — STOP when not grading.
    Run: python scripts/deploy_sagemaker.py --delete
"""
from __future__ import annotations
import argparse
import json
import logging
import time

import boto3
import sagemaker
from sagemaker.huggingface import HuggingFaceModel, get_huggingface_llm_image_uri

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Model configs ──────────────────────────────────────────────────────────
MODELS = {
    "llama2": {
        "model_id": "meta-textgeneration-llama-2-7b-f",
        "model_version": "3.*",
        "hf_model_id": "meta-llama/Llama-2-7b-chat-hf",
        "endpoint_name": "resume-coach-llama2-7b",
        "instance_type": "ml.g5.2xlarge",
        "num_gpus": 1,
        "max_new_tokens": 2048,
        "temperature": 0.1,
    },
    "mistral": {
        "model_id": "huggingface-llm-mistral-7b-instruct",
        "model_version": "1.*",
        "hf_model_id": "mistralai/Mistral-7B-Instruct-v0.2",
        "endpoint_name": "resume-coach-mistral-7b",
        "instance_type": "ml.g5.2xlarge",
        "num_gpus": 1,
        "max_new_tokens": 2048,
        "temperature": 0.1,
    },
}


def deploy_from_jumpstart(model_key: str) -> str:
    cfg = MODELS[model_key]
    session = sagemaker.Session()

    # Load role from env
    import os
    from dotenv import load_dotenv
    load_dotenv()
    role = os.getenv("SAGEMAKER_ROLE_ARN")
    if not role:
        raise ValueError("SAGEMAKER_ROLE_ARN not set in .env")

    logger.info("Deploying %s via JumpStart — instance: %s", cfg["hf_model_id"], cfg["instance_type"])

    from sagemaker.jumpstart.model import JumpStartModel

    model = JumpStartModel(
        model_id=cfg["model_id"],
        model_version=cfg["model_version"],
        sagemaker_session=session,
        role=role,                    # ← add this
    )

    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=cfg["instance_type"],
        endpoint_name=cfg["endpoint_name"],
        model_data_download_timeout=1800,
        container_startup_health_check_timeout=600,
        routing_config={"RoutingStrategy": "LEAST_OUTSTANDING_REQUESTS"},
    )

    logger.info("Endpoint deployed: %s", cfg["endpoint_name"])
    return cfg["endpoint_name"]


def deploy_from_huggingface(model_key: str) -> str:
    """
    Deploy via HuggingFace DLC (Deep Learning Container) with 4-bit quantization.
    Use this if JumpStart model is not available in your region.
    """
    cfg = MODELS[model_key]
    session = sagemaker.Session()
    role = sagemaker.get_execution_role()
    region = session.boto_region_name

    image_uri = get_huggingface_llm_image_uri("huggingface", region=region)
    logger.info("Using DLC image: %s", image_uri)

    env = {
        "HF_MODEL_ID": cfg["hf_model_id"],
        "HF_TASK": "text-generation",
        "SM_NUM_GPUS": str(cfg["num_gpus"]),
        "MAX_INPUT_LENGTH": "4096",
        "MAX_TOTAL_TOKENS": "6144",
        "MAX_BATCH_PREFILL_TOKENS": "4096",
        # 4-bit quantization — reduces memory, fits ml.g5.2xlarge
        "QUANTIZE": "bitsandbytes",
        "LOAD_IN_4BIT": "true",
    }

    model = HuggingFaceModel(
        image_uri=image_uri,
        env=env,
        role=role,
        sagemaker_session=session,
    )

    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=cfg["instance_type"],
        endpoint_name=cfg["endpoint_name"],
        container_startup_health_check_timeout=600,
    )

    logger.info("HuggingFace DLC endpoint deployed: %s", cfg["endpoint_name"])
    return cfg["endpoint_name"]


def test_endpoint(endpoint_name: str, prompt: str = "You are a resume coach. Say hello.") -> None:
    """Quick smoke test of the deployed endpoint."""
    client = boto3.client("sagemaker-runtime")
    payload = {
        "inputs": f"<s>[INST] {prompt} [/INST]",
        "parameters": {"max_new_tokens": 100, "temperature": 0.1},
    }
    response = client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps(payload),
    )
    result = json.loads(response["Body"].read())
    logger.info("Endpoint test response: %s", result)


def delete_endpoint(endpoint_name: str) -> None:
    """Delete the endpoint to stop billing. IMPORTANT: run this after grading."""
    client = boto3.client("sagemaker")
    client.delete_endpoint(EndpointName=endpoint_name)
    logger.info("Endpoint %s deleted. Billing stopped.", endpoint_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["llama2", "mistral"], default="llama2")
    parser.add_argument("--method", choices=["jumpstart", "huggingface"], default="jumpstart")
    parser.add_argument("--delete", action="store_true", help="Delete endpoint")
    parser.add_argument("--test", action="store_true", help="Test existing endpoint")
    args = parser.parse_args()

    if args.delete:
        for key, cfg in MODELS.items():
            try:
                delete_endpoint(cfg["endpoint_name"])
            except Exception as e:
                logger.warning("Could not delete %s: %s", cfg["endpoint_name"], e)
    elif args.test:
        test_endpoint(MODELS[args.model]["endpoint_name"])
    else:
        if args.method == "jumpstart":
            deploy_from_jumpstart(args.model)
        else:
            deploy_from_huggingface(args.model)
