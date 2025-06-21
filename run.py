import warnings
warnings.filterwarnings("ignore")
import argparse
import pandas as pd
from Generation.generate_api import MetGenX
import time

def Convert_generation_dict(generation_dict, output_path):
    rows = []
    for spec_id, (sid, searched_res, types, modified_score, templates_list, candidates) in generation_dict.items():
        row = {
            "SpecID": sid,
            "Type": ";".join(types),
            "Generated_smiles": ";".join([x[0] for x in searched_res]),
            "ModelScore": ';'.join([f"{x[1]:.2f}" for x in searched_res]),
            "CandidateScore":  ';'.join([f"{x:.2f}" for x in modified_score]),
            "Templates_smiles": ";".join([x[0] for x in templates_list]),
            "Templates_score": ";".join([f"{x[1]:.2f}" for x in templates_list]),
            # "Candidates": ";".join(candidates),
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"[INFO] Saved results to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run MetGenX on an spec file.")
    parser.add_argument("--spec_path", type=str, required=True, help="Path to the input .mgf file.")
    parser.add_argument("--polarity", type=str, default="positive", choices=["positive", "negative"], help="Ionization mode.")
    parser.add_argument("--mode", type=str, default="Free", choices=["Free", "Restricted"], help="Generation mode.")
    parser.add_argument("--output", type=str, default="generation_results.csv", help="Path to the output CSV file.")
    parser.add_argument("--db_cutoff", type=float, default=0.4, help="Cutoff threshold for template search.")
    parser.add_argument("--no_rerank", action="store_true", help="Disable reranking.")
    parser.add_argument("--config_path", type=str, default="./weights/generation/config.json")
    parser.add_argument("--config_generation_path", type=str, default="./weights/generation/config_generation.json")
    parser.add_argument("--config_database_path", type=str, default="./weights/generation/config_database.json")

    args = parser.parse_args()

    print(f"[INFO] Initializing MetGenX with polarity={args.polarity} ...")
    metgenx = MetGenX(polarity=args.polarity,
                      config_path=args.config_path,
                      config_generation_path=args.config_generation_path,
                      config_database_path=args.config_database_path)

    print(f"[INFO] Running generation on {args.spec_path} ...")
    start_time = time.time()
    generation_dict = metgenx.generate(
        spec_path=args.spec_path,
        DB_cutoff=args.db_cutoff,
        mode=args.mode,
        rerank=not args.no_rerank
    )

    end_time = time.time()
    time_cost = end_time - start_time
    print(f"[INFO] Generation completed in {time_cost:.2f} seconds.")

    print(f"[INFO] Writing results to CSV ...")
    Convert_generation_dict(generation_dict, args.output)

    print(f"[INFO] All process finished ...")

if __name__ == "__main__":
    main()








