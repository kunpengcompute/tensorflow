import argparse
import importlib
import pkgutil
import traceback
import os

def main():
    parser = argparse.ArgumentParser(description="TF Op Benchmark Framework with KDNN Control")
    parser.add_argument('--op', type=str, help='指定要运行的算子模块名')
    parser.add_argument('--list', action='store_true', help='列出所有模块')
    parser.add_argument('--intra', type=int, default=16, help='intra_op_parallelism_threads, 默认16')
    parser.add_argument('--performance_test', choices=['True', 'False'], default='True', help='是否运行性能测试模块')

    args = parser.parse_args()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' #日志太多, 关掉

    import tensorflow as TF
    from framework.runner import UniversalOpBenchmark
    import ops

    # 4. 扫描模块
    available_ops = {
        name: f'ops.{name}' 
        for loader, name, is_pkg in pkgutil.iter_modules(ops.__path__)
    }

    if args.list:
        print("📁 可用算子列表:")
        for name in available_ops: print(f"  - {name}")
        return

    target_modules = [available_ops[args.op]] if args.op else list(available_ops.values())
    root_log_dir = "bench_logs"
    bench = UniversalOpBenchmark(log_dir=root_log_dir, intra_threads=args.intra)
    
    for module_path in target_modules:
        try:
            module = importlib.import_module(module_path)
            if hasattr(module, 'get_test_cases'):
                cases = module.get_test_cases()
                for case in cases:
                    bench.run_function_test(case)
                if args.performance_test == "True":
                    for case in cases:
                        bench.run_performance_test(case)
        except Exception as e:
            print(f"❌ 运行模块 {module_path} 失败: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()