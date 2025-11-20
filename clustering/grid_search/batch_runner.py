#!/usr/bin/env python3
"""Batch grid-search runner with parallel support."""

import argparse
import os
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Iterable, List, Optional
from tqdm import tqdm
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import time
import signal
import sys

from config import grid_search_output_dir

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 修复导入问题：使用绝对导入而不是相对导入
from clustering.testing.test_superclass import test_adaptive_clustering_on_superclass


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_task_config(
    task_dir: Path,
    args: argparse.Namespace,
    superclasses: List[str],
    k_values: range,
    density_values: range,
    use_parallel: bool,
    max_workers: int,
    parsed_l2_components: Optional[List[str]],
) -> None:
    """将本次批处理任务的参数快照写入 task_dir/task_config.txt。

    内容包含：命令行参数、推导参数（k/density 取值、并行、workers、解析后的 l2_components）、
    超类数量与列表等，便于后续追溯与复现实验。
    """
    lines: List[str] = []
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    lines.append('=' * 80)
    lines.append('Batch Grid Search Task Configuration')
    lines.append('=' * 80)
    lines.append(f'Task Directory: {task_dir.name}')
    lines.append(f'Generated At: {ts}')
    lines.append(f'Number of Superclasses: {len(superclasses)}')
    lines.append(f'Superclasses: {", ".join(superclasses)}')
    lines.append('')

    # 命令行参数
    lines.append('=' * 80)
    lines.append('Command Line Arguments')
    lines.append('=' * 80)
    # 以稳定顺序写出 args.__dict__（注意可能包含 None）
    for key in sorted(vars(args).keys()):
        lines.append(f'{key}: {getattr(args, key)}')
    lines.append('')

    # 推导参数
    lines.append('=' * 80)
    lines.append('Derived Parameters')
    lines.append('=' * 80)
    lines.append(f'k_range: [{k_values.start}, {k_values.stop})')
    lines.append(f'density_range: [{density_values.start}, {density_values.stop}) step={density_values.step}')
    lines.append(f'use_parallel: {use_parallel}')
    lines.append(f'actual_max_workers: {max_workers}')
    lines.append(f'parsed_l2_components: {parsed_l2_components}')
    # 每个超类的 (k, density_percentile) 组合数量
    total_per_superclass = len(k_values) * len(density_values)
    lines.append(f'total_combinations_per_superclass: {total_per_superclass}')
    lines.append('=' * 80)

    cfg_path = task_dir / 'task_config.txt'
    _ensure_dir(task_dir)
    cfg_path.write_text('\n'.join(lines), encoding='utf-8')


def _run_single_process(args_tuple):
    """在单独进程中运行单个参数组合（用于多进程池）"""
    (superclass, model_path, k, density_percentile, eval_version, co_mode,
     co_manual, use_l2, eval_dense, dense_method, assign_model, voting_k,
     use_train_and_test, detail_dense, use_cluster_quality, cluster_distance_method,
     l1_type, separation_weight, penalty_weight, l2_components, l2_component_weights, run_kmeans) = args_tuple

    try:
        import io
        from contextlib import redirect_stdout, redirect_stderr

        captured_output = io.StringIO()

        with redirect_stdout(captured_output), redirect_stderr(captured_output):
            results = test_adaptive_clustering_on_superclass(
                superclass_name=superclass,
                model_path=model_path,
                use_train_and_test=use_train_and_test,
                k=k,
                density_percentile=density_percentile,
                eval_version=eval_version,
                run_kmeans_baseline=run_kmeans,
                co_mode=co_mode,
                co_manual=co_manual,
                use_l2=use_l2,
                eval_dense=eval_dense,
                silent=True,
                dense_method=dense_method,
                assign_model=assign_model,
                voting_k=voting_k,
                detail_dense=detail_dense,
                use_cluster_quality=use_cluster_quality,
                cluster_distance_method=cluster_distance_method,
                l1_type=l1_type,
                separation_weight=separation_weight,
                penalty_weight=penalty_weight,
                l2_components=l2_components,
                l2_component_weights=l2_component_weights,
            )

        return (k, density_percentile, results if isinstance(results, dict) else None)

    except Exception as exc:
        return (k, density_percentile, {'error': str(exc)})


def _run_single(
    superclass: str,
    model_path: str,
    k: int,
    density_percentile: int,
    *,
    eval_version: str,
    co_mode: int,
    co_manual: Optional[float],
    use_l2: bool,
    eval_dense: bool,
    dense_method: int,
    assign_model: int,
    voting_k: int,
    use_train_and_test: bool,
    detail_dense: bool,
    use_cluster_quality: bool,
    cluster_distance_method: int,
    l1_type: str,
    separation_weight: float,
    penalty_weight: float,
    l2_components,
    l2_component_weights,
    run_kmeans: bool,
) -> Optional[dict]:
    """Wrap test_adaptive_clustering_on_superclass with the desired defaults."""
    import io
    from contextlib import redirect_stdout, redirect_stderr

    try:
        # 创建一个缓冲区来捕获所有输出
        captured_output = io.StringIO()

        # 重定向标准输出和标准错误到缓冲区
        with redirect_stdout(captured_output), redirect_stderr(captured_output):
            results = test_adaptive_clustering_on_superclass(
                superclass_name=superclass,
                model_path=model_path,
                use_train_and_test=use_train_and_test,
                k=k,
                density_percentile=density_percentile,
                eval_version=eval_version,
                run_kmeans_baseline=run_kmeans,
                co_mode=co_mode,
                co_manual=co_manual,
                use_l2=use_l2,
                eval_dense=eval_dense,
                silent=True,
                dense_method=dense_method,
                assign_model=assign_model,
                voting_k=voting_k,
                detail_dense=detail_dense,
                use_cluster_quality=use_cluster_quality,
                cluster_distance_method=cluster_distance_method,
                l1_type=l1_type,
                separation_weight=separation_weight,
                penalty_weight=penalty_weight,
                l2_components=l2_components,
                l2_component_weights=l2_component_weights,
            )

        return results if isinstance(results, dict) else None
    except Exception as exc:  # pylint: disable=broad-except
        return {'error': str(exc)}


def run_grid_search(
    superclass: str,
    model_path: str,
    k_values: Iterable[int],
    density_values: Iterable[int],
    *,
    output_dir: Path,
    eval_version: str,
    co_mode: int,
    co_manual: Optional[float],
    use_l2: bool,
    eval_dense: bool,
    dense_method: int,
    assign_model: int,
    voting_k: int,
    use_train_and_test: bool,
    detail_dense: bool,
    use_cluster_quality: bool = False,
    cluster_distance_method: int = 1,
    l1_type: str = 'cross_entropy',
    l2_components=None,
    max_workers: int = None,
    use_parallel: bool = True,
) -> Path:
    """Run grid search for a single superclass and persist results."""
    _ensure_dir(output_dir)
    timestamp = datetime.now().strftime('%m_%d_%H_%M')
    suffix = "_parallel" if use_parallel else ""
    output_file = output_dir / f"{superclass}_{timestamp}{suffix}.txt"

    k_values = list(k_values)
    density_values = list(density_values)
    total = len(k_values) * len(density_values)
    manual_co = co_manual

    if co_mode == 1 and manual_co is None:
        raise ValueError('co_mode=1 requires --co_manual')

    # 确定并行进程数 - 使用一半核心数避免系统过载
    if use_parallel and max_workers is None:
        max_workers = min(cpu_count() // 2, total)  # 使用一半核心数

    if use_parallel:
        return _run_parallel_grid_search(
            superclass, model_path, k_values, density_values, output_file,
            eval_version, co_mode, manual_co, use_l2, eval_dense, dense_method,
            assign_model, voting_k, use_train_and_test, detail_dense,
            use_cluster_quality, cluster_distance_method, l1_type,
            1.0, 1.0, l2_components, None, max_workers
        )
    else:
        return _run_serial_grid_search(
            superclass, model_path, k_values, density_values, output_file,
            eval_version, co_mode, manual_co, use_l2, eval_dense, dense_method,
            assign_model, voting_k, use_train_and_test, detail_dense,
            use_cluster_quality, cluster_distance_method, l1_type,
            1.0, 1.0, l2_components, None
        )


def _run_parallel_grid_search(
    superclass, model_path, k_values, density_values, output_file,
    eval_version, co_mode, manual_co, use_l2, eval_dense, dense_method,
    assign_model, voting_k, use_train_and_test, detail_dense,
    use_cluster_quality, cluster_distance_method, l1_type,
    separation_weight, penalty_weight, l2_components, l2_component_weights, max_workers
):
    """并行版本的网格搜索"""
    total = len(k_values) * len(density_values)

    # 准备参数组合
    param_combinations = list(product(k_values, density_values))
    process_args = []
    for i, (k, density_percentile) in enumerate(param_combinations):
        args_tuple = (
            superclass, model_path, k, density_percentile, eval_version,
            co_mode, manual_co, use_l2, eval_dense, dense_method,
            assign_model, voting_k, use_train_and_test, detail_dense,
            use_cluster_quality, cluster_distance_method, l1_type,
            separation_weight, penalty_weight, l2_components, l2_component_weights,
            i == 0  # 只在第一次运行K-means
        )
        process_args.append(args_tuple)

    # 开始并行执行
    results_dict = {}
    best = None
    kmeans_cache = None
    start_time = time.time()
    interrupted = False

    def signal_handler(signum, frame):
        nonlocal interrupted
        interrupted = True
        print("\n🛑 检测到中断信号，正在安全退出...")
        print("⏳ 等待当前任务完成并清理资源...")

    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)

    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_params = {
                executor.submit(_run_single_process, args): (args[2], args[3])
                for args in process_args
            }

            pbar = tqdm(
                total=total,
                desc=f"Grid search {superclass}",
                unit="param",
                ncols=120,
                position=1,
                leave=False
            )

            for future in as_completed(future_to_params):
                # 检查是否被中断
                if interrupted:
                    pbar.set_description("🛑 正在取消剩余任务...")
                    # 取消所有未完成的任务
                    for f in future_to_params:
                        if not f.done():
                            f.cancel()
                    break

                k, density_percentile = future_to_params[future]

                try:
                    k_result, dp_result, results = future.result()

                    if results and 'error' not in results:
                        results_dict[(k, density_percentile)] = results

                        if kmeans_cache is None and 'kmeans_all_acc' in results:
                            kmeans_cache = {
                                'all_acc': results.get('kmeans_all_acc', 0.0),
                                'old_acc': results.get('kmeans_old_acc', 0.0),
                                'new_acc': results.get('kmeans_new_acc', 0.0),
                                'clusters': results.get('kmeans_n_clusters', 0),
                            }

                        current_acc = results.get('all_acc', 0.0)
                        if best is None or current_acc > best['all_acc']:
                            best = {
                                'params': (k, density_percentile),
                                'all_acc': current_acc,
                            }
                            pbar.set_postfix({
                                'best_k': k,
                                'best_dp': density_percentile,
                                'best_acc': f"{current_acc:.4f}",
                                'status': '🎯NEW BEST'
                            })
                        else:
                            pbar.set_postfix({
                                'k': k,
                                'dp': density_percentile,
                                'acc': f"{current_acc:.4f}",
                                'best_acc': f"{best['all_acc']:.4f}" if best else "N/A"
                            })
                    else:
                        error_msg = results.get('error', 'unknown error') if results else 'no results'
                        pbar.set_postfix({
                            'k': k,
                            'dp': density_percentile,
                            'status': f'FAILED: {error_msg[:20]}...'
                        })

                    pbar.update(1)

                except Exception as exc:
                    pbar.set_postfix({
                        'k': k,
                        'dp': density_percentile,
                        'status': f'ERROR: {str(exc)[:20]}...'
                    })
                    pbar.update(1)

            pbar.close()

    except KeyboardInterrupt:
        interrupted = True
        print("\n🛑 用户中断，正在清理进程池...")

    finally:
        # 恢复默认信号处理
        signal.signal(signal.SIGINT, signal.SIG_DFL)

    elapsed_time = time.time() - start_time

    # 如果被中断，在输出文件中标记
    if interrupted:
        print(f"⚠️  搜索被中断，已完成 {len(results_dict)}/{total} 个参数组合")

    # 写入结果文件
    with output_file.open('w', encoding='utf-8') as handle:
        handle.write(f"Grid search for superclass: {superclass}\n")
        handle.write(f"Model path: {model_path}\n")
        handle.write(f"k values: {k_values}\n")
        handle.write(f"density percentiles: {density_values}\n")
        handle.write(f"l1_type: {l1_type}\n")
        handle.write(f"l2_components: {l2_components}\n")
        handle.write(f"l2_component_weights: {l2_component_weights}\n")
        handle.write(f"Parallel workers: {max_workers}\n")
        handle.write(f"Total time: {elapsed_time:.2f}s\n")
        if len(results_dict) > 0:
            handle.write(f"Avg per combination: {elapsed_time/len(results_dict):.2f}s\n")
        handle.write(f"Success: {len(results_dict)}/{total} combinations\n")
        if interrupted:
            handle.write("⚠️  Search was interrupted by user\n")
        handle.write('=' * 80 + '\n\n')

        # 按照k和density_percentile排序写入结果
        for (k, density_percentile) in sorted(results_dict.keys()):
            results = results_dict[(k, density_percentile)]
            handle.write(f"k={k}, density_percentile={density_percentile}\n")
            handle.write('-' * 80 + '\n')
            handle.write(f"all_acc: {results.get('all_acc', 0.0):.4f}\n")
            handle.write(f"old_acc: {results.get('old_acc', 0.0):.4f}\n")
            handle.write(f"new_acc: {results.get('new_acc', 0.0):.4f}\n")
            handle.write(f"clusters: {results.get('n_clusters', 0)}\n")

            # 添加labeled_acc
            labeled_acc_value = results.get('labeled_acc')
            if labeled_acc_value is not None:
                handle.write(f"labeled_acc: {labeled_acc_value:.4f}\n")
            else:
                handle.write("labeled_acc: N/A\n")

            # 添加L1损失值
            l1_value = results.get('l1')
            if l1_value is not None:
                handle.write(f"l1_loss: {l1_value:.4f}\n")
            else:
                handle.write("l1_loss: N/A\n")

            # 添加聚类质量指标（如果存在）
            loss_dict = results.get('loss_dict', {})
            l2_metrics = loss_dict.get('l2_metrics', {})
            cluster_quality = l2_metrics.get('cluster_quality', {})

            quality_score = cluster_quality.get('quality_score')
            separation_score = cluster_quality.get('separation_score')
            penalty_score = cluster_quality.get('penalty_score')

            if quality_score is not None:
                handle.write(f"quality_score: {quality_score:.4f}\n")
            if separation_score is not None:
                handle.write(f"separation_score: {separation_score:.4f}\n")
            if penalty_score is not None:
                handle.write(f"penalty_score: {penalty_score:.4f}\n")

            components = l2_metrics.get('components', {})
            for comp_name, comp_info in components.items():
                if not isinstance(comp_info, dict):
                    continue
                value = comp_info.get('value')
                contribution = comp_info.get('contribution')
                orientation = comp_info.get('orientation')
                if value is not None:
                    handle.write(f"component_{comp_name}_value: {float(value):.4f}\n")
                if contribution is not None:
                    handle.write(f"component_{comp_name}_contribution: {float(contribution):.4f}\n")
                if orientation is not None:
                    handle.write(f"component_{comp_name}_orientation: {orientation}\n")

            handle.write(f"l2_components: {results.get('l2_components')}\n")
            handle.write(f"l2_component_weights: {results.get('l2_component_weights')}\n")
            handle.write(f"l2_component_params: {results.get('l2_component_params')}\n")

            handle.write('\n')

        handle.write('=' * 80 + '\n')
        if best:
            handle.write(f"Best params: k={best['params'][0]}, density_percentile={best['params'][1]}\n")
            handle.write(f"Best all_acc: {best['all_acc']:.4f}\n")

        if kmeans_cache:
            handle.write(f"K-means baseline: all_acc={kmeans_cache['all_acc']:.4f}\n")

    return output_file


def _run_serial_grid_search(
    superclass, model_path, k_values, density_values, output_file,
    eval_version, co_mode, manual_co, use_l2, eval_dense, dense_method,
    assign_model, voting_k, use_train_and_test, detail_dense,
    use_cluster_quality, cluster_distance_method, l1_type,
    separation_weight, penalty_weight, l2_components, l2_component_weights
):
    """串行版本的网格搜索（原实现）"""
    total = len(k_values) * len(density_values)

    with output_file.open('w', encoding='utf-8') as handle:
        handle.write(f"Grid search for superclass: {superclass}\n")
        handle.write(f"Model path: {model_path}\n")
        handle.write(f"k values: {k_values}\n")
        handle.write(f"density percentiles: {density_values}\n")
        handle.write(f"eval_version: {eval_version}\n")
        handle.write(f"co_mode: {co_mode}\n")
        handle.write(f"co_manual: {manual_co}\n")
        handle.write(f"use_l2: {use_l2}\n")
        handle.write(f"eval_dense: {eval_dense}\n")
        handle.write(f"dense_method: {dense_method}\n")
        handle.write(f"assign_model: {assign_model}\n")
        handle.write(f"voting_k: {voting_k}\n")
        handle.write(f"use_train_and_test: {use_train_and_test}\n")
        handle.write(f"detail_dense: {detail_dense}\n")
        handle.write(f"use_cluster_quality: {use_cluster_quality}\n")
        handle.write(f"cluster_distance_method: {cluster_distance_method}\n")
        handle.write(f"l1_type: {l1_type}\n")
        handle.write(f"l2_components: {l2_components}\n")
        handle.write(f"l2_component_weights: {l2_component_weights}\n")
        handle.write('=' * 80 + '\n\n')

        best = None
        kmeans_cache = None
        param_combinations = list(product(k_values, density_values))

        pbar = tqdm(
            param_combinations,
            desc=f"Grid search for {superclass}",
            unit="param",
            ncols=120,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
            position=1,
            leave=False
        )

        for index, (k, density_percentile) in enumerate(pbar, start=1):
            pbar.set_postfix({
                'k': k,
                'dp': density_percentile,
                'best_acc': f"{best['all_acc']:.4f}" if best else "N/A"
            })

            handle.write(f"k={k}, density_percentile={density_percentile}\n")
            handle.write('-' * 80 + '\n')

            results = _run_single(
                superclass, model_path, k, density_percentile,
                eval_version=eval_version, co_mode=co_mode, co_manual=manual_co,
                use_l2=use_l2, eval_dense=eval_dense, dense_method=dense_method,
                assign_model=assign_model, voting_k=voting_k,
                use_train_and_test=use_train_and_test, detail_dense=detail_dense,
                use_cluster_quality=use_cluster_quality,
                cluster_distance_method=cluster_distance_method,
                l1_type=l1_type,
                separation_weight=separation_weight,
                penalty_weight=penalty_weight,
                l2_components=l2_components,
                l2_component_weights=l2_component_weights,
                run_kmeans=kmeans_cache is None,
            )

            if not results or 'error' in results:
                handle.write(f"FAILED: {results.get('error', 'unknown error')}\n\n")
                pbar.set_postfix({
                    'k': k, 'dp': density_percentile, 'status': 'FAILED',
                    'best_acc': f"{best['all_acc']:.4f}" if best else "N/A"
                })
                continue

            if kmeans_cache is None:
                kmeans_cache = {
                    'all_acc': results.get('kmeans_all_acc', 0.0),
                    'old_acc': results.get('kmeans_old_acc', 0.0),
                    'new_acc': results.get('kmeans_new_acc', 0.0),
                    'clusters': results.get('kmeans_n_clusters', 0),
                }
            else:
                results['kmeans_all_acc'] = kmeans_cache['all_acc']
                results['kmeans_old_acc'] = kmeans_cache['old_acc']
                results['kmeans_new_acc'] = kmeans_cache['new_acc']
                results['kmeans_n_clusters'] = kmeans_cache['clusters']

            current_acc = results.get('all_acc', 0.0)
            handle.write(f"all_acc: {current_acc:.4f}\n")
            handle.write(f"old_acc: {results.get('old_acc', 0.0):.4f}\n")
            handle.write(f"new_acc: {results.get('new_acc', 0.0):.4f}\n")
            handle.write(f"clusters: {results.get('n_clusters', 0)}\n")

            # 添加labeled_acc
            labeled_acc_value = results.get('labeled_acc')
            if labeled_acc_value is not None:
                handle.write(f"labeled_acc: {labeled_acc_value:.4f}\n")
            else:
                handle.write("labeled_acc: N/A\n")

            # 添加L1损失值
            l1_value = results.get('l1')
            if l1_value is not None:
                handle.write(f"l1_loss: {l1_value:.4f}\n")
            else:
                handle.write("l1_loss: N/A\n")

            # 添加聚类质量指标（如果存在）
            loss_dict = results.get('loss_dict', {})
            l2_metrics = loss_dict.get('l2_metrics', {})
            cluster_quality = l2_metrics.get('cluster_quality', {})

            quality_score = cluster_quality.get('quality_score')
            separation_score = cluster_quality.get('separation_score')
            penalty_score = cluster_quality.get('penalty_score')

            if quality_score is not None:
                handle.write(f"quality_score: {quality_score:.4f}\n")
            if separation_score is not None:
                handle.write(f"separation_score: {separation_score:.4f}\n")
            if penalty_score is not None:
                handle.write(f"penalty_score: {penalty_score:.4f}\n")

            components = l2_metrics.get('components', {})
            for comp_name, comp_info in components.items():
                if not isinstance(comp_info, dict):
                    continue
                value = comp_info.get('value')
                contribution = comp_info.get('contribution')
                orientation = comp_info.get('orientation')
                if value is not None:
                    handle.write(f"component_{comp_name}_value: {float(value):.4f}\n")
                if contribution is not None:
                    handle.write(f"component_{comp_name}_contribution: {float(contribution):.4f}\n")
                if orientation is not None:
                    handle.write(f"component_{comp_name}_orientation: {orientation}\n")

            handle.write('\n')

            if best is None or current_acc > best['all_acc']:
                best = {'params': (k, density_percentile), 'all_acc': current_acc}
                pbar.set_postfix({
                    'k': k, 'dp': density_percentile, 'acc': f"{current_acc:.4f}",
                    'best_acc': f"{current_acc:.4f}", 'status': '🎯NEW BEST'
                })
            else:
                pbar.set_postfix({
                    'k': k, 'dp': density_percentile, 'acc': f"{current_acc:.4f}",
                    'best_acc': f"{best['all_acc']:.4f}"
                })

        pbar.close()
        print(f"\r{' ' * 120}\r", end='', flush=True)

        handle.write('=' * 80 + '\n')
        handle.write(f"Completed {index if total else 0}/{total} combinations\n")
        if best:
            handle.write(f"Best params: k={best['params'][0]}, density_percentile={best['params'][1]}\n")
            handle.write(f"Best all_acc: {best['all_acc']:.4f}\n")

    return output_file


def parse_superclasses(defaults: List[str], override: Optional[str], file_path: Optional[str]) -> List[str]:
    if override:
        return [item.strip() for item in override.split(',') if item.strip()]
    if file_path:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f'superclass list file not found: {file_path}')
        return [line.strip() for line in path.read_text(encoding='utf-8').splitlines() if line.strip() and not line.startswith('#')]
    return defaults


DEFAULT_SUPERCLASSES = [
    "trees",
    "humans",
    "vehicles",
    "buildings",
]

# ALL_15_SUPERCLASSES 列表来源：data/cifar100_superclass.py 中 CIFAR100_SUPERCLASSES 定义。
# 为降低耦合避免导入失败，这里硬编码；如上游变动请同步更新。
ALL_15_SUPERCLASSES = [
    "trees",
    "flowers",
    "fruits_vegetables",
    "mammals",
    "marine_animals",
    "insects_arthropods",
    "reptiles",
    "humans",
    "furniture",
    "containers",
    "vehicles",
    "electronic_devices",
    "buildings",
    "terrain",
    "weather_phenomena",
]


def main() -> None:
    parser = argparse.ArgumentParser(description='Python-based batch grid search runner.')
    parser.add_argument('--model_path', type=str, default=None,
                        help='预训练模型检查点路径（如 /path/to/model.pth）。默认None表示优先使用缓存的特征文件；首次运行需提供模型以提取特征。')
    parser.add_argument('--output_dir', type=str, default=grid_search_output_dir,
                        help='搜索结果输出目录（包含每个超类的 *.txt 结果文件）')
    parser.add_argument('--superclasses', type=str, default=None,
                        help='逗号分隔的超类列表（例如: trees,humans）。若未提供，将使用默认集合或 --superclass_file 指定的列表。')
    parser.add_argument('--superclass_file', type=str, default=None,
                        help='可选：文本文件路径，每行一个超类名称；以 # 开头的行为注释。')
    parser.add_argument('--k_min', type=int, default=3,
                        help='KNN 图的最小 k 值（默认3）。k 控制近邻数量，影响图连通性与稳健性。')
    parser.add_argument('--k_max', type=int, default=21,
                        help='KNN 图的最大 k 值上界（默认21，半开区间）。')
    parser.add_argument('--density_min', type=int, default=40,
                        help='密度阈值的最小百分位（默认40）。')
    parser.add_argument('--density_max', type=int, default=100,
                        help='密度阈值的最大百分位（默认100，不含）。')
    parser.add_argument('--density_step', type=int, default=5,
                        help='密度阈值步长（默认5）。')
    parser.add_argument('--co_mode', type=int, default=2, choices=[1, 2, 3],
                        help='cutoff 模式：1=手动（需 --co_manual），2=自动（推荐），3=无 cutoff。')
    parser.add_argument('--co_manual', type=float, default=None,
                        help='co_mode=1 时的手动 cutoff 值。')
    parser.add_argument('--eval_version', type=str, default='v2', choices=['v1', 'v2'],
                        help='评估版本（默认v2）。')
    parser.add_argument('--dense_method', type=int, default=0, choices=[0, 1, 2, 3],
                        help='密集样本选择方法：0=关闭，1=局部密度，2=全局密度，3=混合。')
    parser.add_argument('--assign_model', type=int, default=2, choices=[1, 2, 3],
                        help='样本分配模型：1=最近质心，2=KNN 投票（默认），3=软分配。')
    parser.add_argument('--voting_k', type=int, default=5,
                        help='KNN 投票的 k（默认5）。')
    parser.add_argument('--use_l2', action='store_true')
    parser.add_argument('--no_l2', dest='use_l2', action='store_false')
    parser.set_defaults(use_l2=True)
    parser.add_argument('--eval_dense', action='store_true')
    parser.add_argument('--use_train_and_test', action='store_true', default=True)
    parser.add_argument('--detail_dense', action='store_true')

    # 聚类质量评估参数
    parser.add_argument('--use_cluster_quality', action='store_true', default=False,
                        help='是否使用聚类质量评估指标作为L2损失')
    parser.add_argument('--cluster_distance_method', type=int, default=1, choices=[1, 2, 3],
                        help='簇距离计算方法：1=最近k对点平均距离，2=所有点对平均距离，3=原型距离')
    parser.add_argument('--l1_type', type=str, default='cross_entropy', choices=['accuracy', 'cross_entropy'],
                        help='L1监督损失类型：accuracy=基于匈牙利算法的准确率损失(1-ACC)，cross_entropy=基于簇类别分布的交叉熵损失（默认）')

    # L2组件选择（仅用于数据收集阶段，不涉及权重配置）
    parser.add_argument('--l2_components', type=str, default=None,
                        help='指定要计算的 L2 组件（空格或逗号分隔）。可选: separation, penalty, silhouette。示例: "separation silhouette"。')

    # 并行相关参数
    parser.add_argument('--max_workers', type=int, default=None,
                        help='并行进程数上限（默认自动=CPU核心数的一半且不超过组合数）。')
    parser.add_argument('--no_parallel', action='store_true',
                        help='禁用并行，改为串行执行（便于调试或内存受限环境）。')
    parser.add_argument('--count_all', action='store_true', default=False,
                        help='启用全部15个自定义超类（优先级最高，覆盖 --superclasses 与 --superclass_file）。 '
                             '包含: trees, flowers, fruits_vegetables, mammals, marine_animals, insects_arthropods, '
                             'reptiles, humans, furniture, containers, vehicles, electronic_devices, buildings, terrain, weather_phenomena。')

    args = parser.parse_args()

    def _parse_l2_components(value: str):
        if value is None:
            return None
        return [part.strip() for part in value.replace(',', ' ').split() if part.strip()]

    parsed_l2_components = _parse_l2_components(args.l2_components)

    if args.count_all:
        superclasses = ALL_15_SUPERCLASSES
        names = ', '.join(superclasses)
        print(f"ℹ️  启用全部 {len(superclasses)} 个超类（--count_all）")
        print(f"   超类列表: {names}")
    else:
        superclasses = parse_superclasses(DEFAULT_SUPERCLASSES, args.superclasses, args.superclass_file)
    output_root = Path(args.output_dir)

    k_values = range(args.k_min, args.k_max)
    density_values = range(args.density_min, args.density_max, args.density_step)

    # 确定每个超类的并行进程数
    use_parallel = not args.no_parallel
    total_combinations = len(k_values) * len(density_values)

    if use_parallel:
        if args.max_workers:
            max_workers = min(args.max_workers, total_combinations)
        else:
            max_workers = min(cpu_count() // 2, total_combinations)  # 使用一半核心数
    else:
        max_workers = 1

    # 创建任务级目录（按超类数量与时间戳命名）并写入配置
    timestamp = datetime.now().strftime('%m_%d_%H_%M')
    task_dirname = f"{len(superclasses)}class_{timestamp}"
    task_dir = output_root / task_dirname
    _ensure_dir(task_dir)
    _write_task_config(
        task_dir=task_dir,
        args=args,
        superclasses=superclasses,
        k_values=k_values,
        density_values=density_values,
        use_parallel=use_parallel,
        max_workers=max_workers,
        parsed_l2_components=parsed_l2_components,
    )

    # 添加超类级别的进度条
    superclass_pbar = tqdm(
        superclasses,
        desc="Processing superclasses",
        unit="superclass",
        position=0,
        leave=True,
        ncols=120
    )

    for superclass in superclass_pbar:
        superclass_pbar.set_description(f"Processing {superclass}")

        output_file = run_grid_search(
            superclass,
            args.model_path,
            k_values,
            density_values,
            output_dir=task_dir / superclass,
            eval_version=args.eval_version,
            co_mode=args.co_mode,
            co_manual=args.co_manual,
            use_l2=args.use_l2,
            eval_dense=args.eval_dense,
            dense_method=args.dense_method,
            assign_model=args.assign_model,
            voting_k=args.voting_k,
            use_train_and_test=args.use_train_and_test,
            detail_dense=args.detail_dense,
            use_cluster_quality=args.use_cluster_quality,
            cluster_distance_method=args.cluster_distance_method,
            l1_type=args.l1_type,
            l2_components=parsed_l2_components,
            max_workers=max_workers,
            use_parallel=use_parallel,
        )

        superclass_pbar.set_postfix_str(f"Saved: {output_file.name}")

    superclass_pbar.close()

    if use_parallel:
        mode_str = f"parallel ({max_workers} workers)"
    else:
        mode_str = "serial"

    print("\n🎉 Batch grid search completed!")
    print(f"📁 Task directory: {task_dir}")
    print(f"🧾 Config file: {task_dir / 'task_config.txt'}")
    print(f"💡 下游 L2/L1L2 工具请指定 --search_dir {task_dir}")
    print(f"📊 Processed {len(superclasses)} superclasses ({mode_str})")


if __name__ == '__main__':
    # 设置多进程启动方法
    mp.set_start_method('spawn', force=True)
    main()
