# AI-Immune-System-Challenge

## Brief

This competition is the world’s first practical challenge toward realizing an AI Immune System—a biologically inspired framework in which multiple AI agents monitor one another to detect and control abnormal or dangerous behavior. Just as the human immune system identifies threats the body itself cannot consciously perceive, an AI Immune System aims to surface risks that may be invisible or unintuitive to humans.

Participants will build machine learning models to identify dangerous or unsafe statements embedded within AI agent conversations, where harmful intent may be obscured by natural-sounding or indirect language. These conversations are designed to reflect realistic machine-to-machine interactions in which risk does not appear as explicit commands or overtly malicious content.

This task captures a core challenge of modern AI safety: detecting threats that evade simple rules, keyword filters, or direct interpretation, and that may not be immediately recognizable even to human reviewers. Successful solutions must go beyond surface-level text classification and instead uncover deeper statistical, semantic, or structural signals.

By participating, you will tackle a cutting-edge AI safety problem, develop techniques applicable to real-world autonomous systems, and contribute to foundational technology for building self-monitoring, resilient, and trustworthy AI ecosystems.

## Data Overview

The goal of this competition is to predict whether each text in jsonl files is harmful conversation or not, which is indicated by "label" column (“TRUE”: Harmful conversation “FALSE”: Non-harmful conversation).

Downloadable file "ai-immune..zip" includes the following files:

1. train_labeled_comp.jsonl: file to train your machine learning model.

2. test_labeled_comp.jsonl: file that can be used to test how well your model performs on unseen data. This is the file you're going to make predictions on with your trained model and create a submission file.

3. solution_format.csv: example of the format that the submission file needs to be in to be properly scored.

## Rules

1. Terms of Participation

This competition is governed by the following Terms of Participation. Participants must agree to and comply with these Terms in order to participate.

2. Submission Limits

Users may make a maximum of ten submissions per day. If a user wishes to submit additional files after reaching this limit, they must wait until the following day. Please keep this limitation in mind when uploading a submission.csv file. Any attempt to circumvent the stated limits will result in disqualification.

3. External Data and Pre-trained Models

- External Datasets: The use of external datasets (e.g., additional training samples or labels from other sources) is strictly prohibited.
- Pre-trained Models: The use of publicly available, open-source pre-trained models (e.g., BERT, RoBERTa, Llama, etc.) and Embedding models is permitted, as they are considered part of the model architecture.
- Proprietary APIs: The use of commercial or proprietary APIs (e.g., OpenAI GPT series, Claude, Gemini API) is strictly prohibited due to reproducibility constraints.

4. Computation Resource Limits

To ensure that all solutions can be verified by the bitgrit team, the submitted code must be executable within the following hardware constraints. Any submission that fails to run due to resource exhaustion (Out of Memory) will be disqualified.

- RAM: Maximum 32 GB
- VRAM: Maximum 16 GB (Equivalent to 1x NVIDIA T4 GPU)

5. Dataset Distribution

Uploading the competition dataset to other websites is strictly prohibited. Users who do not comply with this rule will be disqualified.

6. Prize Award and Verification Requirements

A competition prize will be awarded only after the submitted code and solution have been received, successfully executed, and verified for validity. Once winners are announced and contacted, they must provide the following by MM DD, 2026 in order to qualify as a competition winner and receive their prize:

- All source files required to preprocess the data.
- All source files required to build, train, and generate predictions using the processed data.
- Model Weights: The actual model weights used or a permanent link to the specific version of the pre-trained model utilized.
- A requirements.txt (or equivalent) file listing all required libraries and their versions.
- A README file containing:
- Clear, unambiguous instructions to reproduce the predictions from start to finish, including data preprocessing, feature extraction, model training, and prediction generation.
- Environment details where the model was developed and trained, including operating system, memory (RAM), disk space, CPU/GPU used, and any required environment configurations.
- Clear answers to the following questions: Which data files are being used? How are these files processed? What algorithm is used and what are its main hyperparameters? Any additional comments relevant to understanding and using the model.

If these materials are not provided or do not meet the minimum requirements listed above, the prize cannot be awarded.

7. Reproducibility of Results

- Determinism: Participants must fix all random seeds and set the inference temperature to 0 (where applicable) to ensure reproducible results.
- Score Consistency: The submitted solution should ideally generate the same output that produced the leaderboard score. If the score obtained during verification differs slightly due to the non-deterministic nature of certain hardware/software stacks, the result may still be accepted at the organizers' discretion, provided the logic remains consistent and the score is an approximation of the original.

8. Final Decisions

All prize awards are subject to verification of eligibility and compliance with these Terms of Participation. All decisions made by bitgrit and the Competition Sponsor are final and binding.

9. Taxes

Prize payments may be subject to local, state, federal, and foreign tax reporting and withholding requirements.

10. Tie-Breaking Rule

If two or more participants achieve the same score on the leaderboard, the participant who submitted the winning file first will be considered the winner.

11. Individual Participation Only

All submissions must be made by individuals; team submissions are not allowed. Users who violate this rule will be immediately disqualified if identical or very similar scores and/or solutions are identified.

12. Data Deletion Requirement

Participants must delete all Company-Provided Information immediately after the completion of the competition.

13. Contact Information

For any questions regarding this competition, please contact us at info@bitgrit.com.
