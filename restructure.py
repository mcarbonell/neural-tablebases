import os
import shutil
import glob

def main():
    src_dir = "src"
    scripts_dir = "scripts"
    legacy_dir = "src/legacy"
    data_dir = "src/data"
    model_dir = "src/model"
    search_dir = "src/search"
    eval_dir = "src/evaluation"

    for d in [scripts_dir, legacy_dir, data_dir, model_dir, search_dir, eval_dir]:
        os.makedirs(d, exist_ok=True)
        if d.startswith("src/"):
            open(os.path.join(d, "__init__.py"), "w").close()

    file_moves = {
        "analyze_fen.py": scripts_dir,
        "check_dtz_progress.py": scripts_dir,
        "check_endgame_inventory.py": scripts_dir,
        "check_fens_depth.py": scripts_dir,
        "find_dtz_errors.py": scripts_dir,
        "find_dtz_outliers.py": scripts_dir,
        "find_errors.py": scripts_dir,
        "test_dml_ops.py": scripts_dir,
        "test_npu_inference.py": scripts_dir,
        "gnn_utils.py": scripts_dir,
        
        "generate_all_v6.py": legacy_dir,
        "generate_all_v7.py": legacy_dir,
        "validate_v6.py": legacy_dir,
        "verify_v6_search.py": legacy_dir,
        "train_gnn_proto.py": legacy_dir,
        "train_v8_prototype.py": legacy_dir,
        "train_large_scale.py": legacy_dir,
        "canonical_basic.py": legacy_dir,
        "canonical_forms_fixed.py": legacy_dir,
        "train.py": legacy_dir,
        "models.py": legacy_dir,
        "generate_all_endgames.py": legacy_dir,
        "generate_datasets.py": legacy_dir,
        "generate_datasets_canonical.py": legacy_dir,
        "generate_datasets_parallel.py": legacy_dir,
        "generate_datasets_parallel_canonical.py": legacy_dir,
        
        "dataset_v8.py": data_dir,
        "build_universal_shards.py": data_dir,
        "encoding_gnn.py": data_dir,
        "encoding_invariant.py": data_dir,
        "canonical_forms.py": data_dir,
        
        "models_v8.py": model_dir,
        
        "searcher_v8.py": search_dir,
        "rust_engine.py": search_dir,
        "search_poc.py": search_dir,
        "verify_search_correction.py": search_dir,
        
        "eval_v8_tablebase.py": eval_dir,
        "evaluate_model.py": eval_dir,
        "export_onnx.py": eval_dir,
        "export_onnx_v8.py": eval_dir,
    }

    moved_count = 0
    for f, dest in file_moves.items():
        src_path = os.path.join(src_dir, f)
        if os.path.exists(src_path):
            shutil.move(src_path, os.path.join(dest, f))
            moved_count += 1

    for d in ["rust_movegen", "rust-movegen_original"]:
        src_path = os.path.join(src_dir, d)
        if os.path.exists(src_path):
            shutil.move(src_path, os.path.join(search_dir, d))

    old_train = os.path.join(src_dir, "train_v8.py")
    if os.path.exists(old_train):
        shutil.move(old_train, os.path.join(src_dir, "train.py"))

    old_gen = os.path.join(src_dir, "generate_gnn_dataset.py")
    if os.path.exists(old_gen):
        shutil.move(old_gen, os.path.join(src_dir, "generate_data.py"))

    import_fixes = [
        ("from models_v8 import", "from model.models_v8 import"),
        ("import models_v8", "from model import models_v8"),
        
        ("from dataset_v8 import", "from data.dataset_v8 import"),
        ("import dataset_v8", "from data import dataset_v8"),
        
        ("from encoding_gnn import", "from data.encoding_gnn import"),
        ("import encoding_gnn", "from data import encoding_gnn"),
        
        ("from canonical_forms import", "from data.canonical_forms import"),
        ("import canonical_forms", "from data import canonical_forms"),
        
        ("from encoding_invariant import", "from data.encoding_invariant import"),
        ("import encoding_invariant", "from data import encoding_invariant"),
        
        ("from searcher_v8 import", "from search.searcher_v8 import"),
        ("import searcher_v8", "from search import searcher_v8"),
        
        ("from rust_engine import", "from search.rust_engine import"),
        ("import rust_engine", "from search import rust_engine"),
        
        ("from models import", "from legacy.models import"),
        ("import models", "from legacy import models"),
        
        ("from train import", "from legacy.train import"),
        ("import train", "from legacy import train"),

        ("from generate_datasets import", "from legacy.generate_datasets import"),
        ("import generate_datasets", "from legacy import generate_datasets"),
        
        ("from generate_datasets_parallel import", "from legacy.generate_datasets_parallel import"),
        ("import generate_datasets_parallel", "from legacy import generate_datasets_parallel"),
    ]

    py_files = []
    for root, _, files in os.walk(src_dir):
        for f in files:
            if f.endswith(".py"):
                py_files.append(os.path.join(root, f))
    for root, _, files in os.walk(scripts_dir):
        for f in files:
            if f.endswith(".py"):
                py_files.append(os.path.join(root, f))
                
    py_files.append("run_tests.py")
    if os.path.exists("tests/test_v8_pipeline.py"):
        py_files.append("tests/test_v8_pipeline.py")
    
    modified_count = 0
    for pyf in py_files:
        if not os.path.exists(pyf): continue
        with open(pyf, "r", encoding="utf-8") as f:
            content = f.read()
        
        orig_content = content
        for old, new in import_fixes:
            content = content.replace(old, new)
            
        if content != orig_content:
            with open(pyf, "w", encoding="utf-8") as f:
                f.write(content)
            modified_count += 1

    print(f"Moved {moved_count} files, Modified imports in {modified_count} files.")

if __name__ == "__main__":
    main()
