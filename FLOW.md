```bash
1. config.yml  = very imp (brain)
2. inference.py  = takes load model(pkl, scaler), input, predicts and store op .... this file is for deployment.py as entry point
3. aws_utils.py  =  loads the config.yml and assigns my code a role(the same one we made on aws). cos for our code to work, the code should have a role.
4. data_processing.py
5. mlflow  = starting mlflow
6. model.py  = train function the model
7. train_local.py  = this is the orchestrator, where all the script comes together
8. in train_local.py we are storing our metrics/artifacts/params in numbered folder in mlruns (bydefault behaviour to go in numbered folder)
9. we also create our temporary folder for models for our sake


to run on sagemaker = python train_sagemaker.py --instance-type ml.m5.large (traning job)
to deploy = python deploy_model_copy.py --model-path s3://ashish-sagemaker-runs-2026/output/ml-training-1777187853/output/model.tar.gz