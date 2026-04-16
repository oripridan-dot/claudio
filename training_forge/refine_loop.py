import os
import subprocess
import json

def run_training_cycle(epochs):
    print(f"\n--- Starting Training Cycle ({epochs} epochs) ---")
    subprocess.run(["python", "train.py", "--epochs", str(epochs), "--batch-size", "16"], check=True)

def run_evaluation():
    print("\n--- Running Evaluation Synthesis ---")
    subprocess.run(["python", "evaluate.py"], check=True)

def run_fidelity_analyzer():
    print("\n--- Running Fidelity Analysis ---")
    subprocess.run(["python", "fidelity_analyzer.py"], check=True)
    with open("fidelity_metrics.json", "r") as f:
        return json.load(f)

def execute_refinement_loop(target_mcd=10.0, target_lsd=3.0, max_iterations=5):
    print("========================================")
    print("INITIATING AI FIDELITY REFINEMENT LOOP")
    print("========================================")
    
    for iteration in range(max_iterations):
        print(f"\n>>> ITERATION {iteration + 1} / {max_iterations} <<<")
        
        # 1. Train
        # We start with 12 epochs per refinement pass
        run_training_cycle(epochs=12)
        
        # 2. Evaluate
        run_evaluation()
        
        # 3. Analyze
        metrics = run_fidelity_analyzer()
        
        mcd = metrics.get("MCD", float('inf'))
        lsd = metrics.get("LSD", float('inf'))
        rmse = metrics.get("F0_RMSE", float('inf'))
        
        print(f"\n[Iteration {iteration + 1} Results]")
        print(f"MCD:  {mcd:.4f} (Target < {target_mcd})")
        print(f"LSD:  {lsd:.4f}  (Target < {target_lsd})")
        print(f"F0 RMSE: {rmse:.4f}")
        
        # 4. Check Convergence
        if mcd < target_mcd and lsd < target_lsd:
            print("\n✅ SUPREME FIDELITY ACHIEVED! Targets met.")
            break
        else:
            print("\n⚠️ Targets not yet met. Commencing next learning cycle to refine weights...")
            # Here one could dynamically read pyproject or adjust train.py LR, 
            # but standard Adam convergence via checkpoint continuation takes care of this optimally.

if __name__ == "__main__":
    execute_refinement_loop()
