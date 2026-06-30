import json
from pathlib import Path

DATA_DIR = Path("/Users/anshrohilla/Documents/Verbalyze/scratch/domain_specific_data_v4/as")
input_file = DATA_DIR / "transcriptsv1_as.jsonl"
main_output = DATA_DIR / "transcriptsv1_as_with_audio.jsonl"
part1_file = DATA_DIR / "transcriptsv1_as_with_audio_part1.jsonl"
part2_file = DATA_DIR / "transcriptsv1_as_with_audio_part2.jsonl"

def main():
    print("Starting Assamese partitions merge...")
    
    # 1. Load the original order of IDs
    if not input_file.exists():
        print(f"Error: Input file {input_file} not found.")
        return
        
    order_map = {}
    with open(input_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if line.strip():
                row = json.loads(line)
                order_map[row["id"]] = idx
                
    print(f"Loaded original order for {len(order_map)} records.")
    
    # 2. Gather all generated records
    all_merged_records = {}
    
    # Load main file records
    if main_output.exists():
        with open(main_output, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    all_merged_records[row["id"]] = row
        print(f"Loaded {len(all_merged_records)} records from main output file.")
        
    # Load part1 records
    if part1_file.exists():
        part1_count = 0
        with open(part1_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    all_merged_records[row["id"]] = row
                    part1_count += 1
        print(f"Loaded {part1_count} records from part1 partition.")
        
    # Load part2 records
    if part2_file.exists():
        part2_count = 0
        with open(part2_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    all_merged_records[row["id"]] = row
                    part2_count += 1
        print(f"Loaded {part2_count} records from part2 partition.")
        
    total_records = len(all_merged_records)
    print(f"Total unique records gathered: {total_records} out of {len(order_map)}")
    
    # 3. Sort records by original order
    sorted_records = []
    missing_ids = []
    for row_id, idx in sorted(order_map.items(), key=lambda x: x[1]):
        if row_id in all_merged_records:
            sorted_records.append(all_merged_records[row_id])
        else:
            missing_ids.append(row_id)
            
    if missing_ids:
        print(f"Warning: {len(missing_ids)} records are missing from the merged output!")
        print(f"Missing IDs sample: {missing_ids[:10]}")
    else:
        print("Success: All records are accounted for and merged!")
        
    # 4. Write back to main output file
    temp_output = DATA_DIR / "transcriptsv1_as_with_audio_merged.jsonl"
    with open(temp_output, "w", encoding="utf-8") as f:
        for row in sorted_records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            
    print(f"Wrote merged output to {temp_output}")
    
    # Backup and replace
    backup_path = DATA_DIR / "transcriptsv1_as_with_audio_pre_merge.jsonl"
    if main_output.exists():
        main_output.rename(backup_path)
        print(f"Backed up pre-merge main file to {backup_path}")
        
    temp_output.rename(main_output)
    print(f"Successfully replaced main output file with the merged version!")

if __name__ == "__main__":
    main()
