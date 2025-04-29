import subprocess

print("=== Step 1: Verifying .npy Dataset ===")
subprocess.run(["python", "ssl_preprocess.py"])

print("\n=== Step 2: Training Semi-Supervised EEG Model ===")
subprocess.run(["python", "training/train_ssl.py"])

print("\n=== Step 3: Evaluating Model on Test Data ===")
subprocess.run(["python", "evaluation/eval_ssl.py"])

print("\n=== All Steps Completed ===")
print("=== Check the output files in the current directory ===")
print("=== End of Process ===") 