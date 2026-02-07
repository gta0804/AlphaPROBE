import concurrent.futures
import threading
import time
from tqdm import tqdm
from collections import defaultdict, deque

class NonBlockingThreadPool:
    """
    一个基于 concurrent.futures.ThreadPoolExecutor 的高级线程池。
    它封装了任务提交、结果收集和进度条显示，并增加了对工作线程状态的实时监控，
    包括成功/失败任务数以及任务执行耗时。
    特别适合处理大规模数据集。
    """
    def __init__(self, max_workers, moving_avg_window=20):
        """
        初始化线程池。
        Args:
            max_workers (int): 最大工作线程数。
            moving_avg_window (int): 用于计算移动平均耗时的任务窗口大小。
        """
        self.max_workers = max_workers
        self.MA_WINDOW_SIZE = moving_avg_window
        
        # 任务计数器
        self._thread_success_counts = defaultdict(int)
        self._thread_failure_counts = defaultdict(int)
        
        # 新增：任务耗时跟踪器
        self._thread_last_duration = defaultdict(float)
        self._thread_duration_history = defaultdict(lambda: deque(maxlen=self.MA_WINDOW_SIZE))
        
        # 单个锁保护所有共享状态
        self._lock = threading.Lock()

    def _task_wrapper(self, func, *args, **kwargs):
        """
        一个内部包装器，用于执行原始任务并记录所有监控指标。
        """
        thread_id = threading.get_ident()
        start_time = time.monotonic()
        
        try:
            result = func(*args, **kwargs)
            # 任务成功，更新成功计数
            with self._lock:
                self._thread_success_counts[thread_id] += 1
            return result
        except Exception as e:
            # 任务失败，更新失败计数
            with self._lock:
                self._thread_failure_counts[thread_id] += 1
            raise e
        finally:
            # 无论成功或失败，都记录耗时
            duration = time.monotonic() - start_time
            with self._lock:
                self._thread_last_duration[thread_id] = duration
                self._thread_duration_history[thread_id].append(duration)

    def _print_status(self, start_time):
        """
        美观地打印每个线程的完整状态，包括计数和耗时。
        """
        with self._lock:
            # 复制所有需要的数据，以尽快释放锁
            success_counts = self._thread_success_counts.copy()
            failure_counts = self._thread_failure_counts.copy()
            last_durations = self._thread_last_duration.copy()
            duration_histories = self._thread_duration_history.copy()
        
        elapsed_time = time.time() - start_time
        all_thread_ids = set(success_counts.keys()) | set(failure_counts.keys())
        
        header = (f"| {'线程ID':<20} | {'成功':<8} | {'失败':<8} | "
                  f"{'上次耗时(s)':<15} | {'平均耗时(s)':<15} |")
        separator = f"|{'-'*22}|{'-'*10}|{'-'*10}|{'-'*17}|{'-'*17}|"
        
        output_lines = [
            f"\n--- 线程池状态 (已运行 {elapsed_time:.2f} 秒) ---",
            header,
            separator
        ]
        
        for thread_id in sorted(all_thread_ids):
            s_count = success_counts.get(thread_id, 0)
            f_count = failure_counts.get(thread_id, 0)
            last_dur = last_durations.get(thread_id, 0.0)
            
            # 计算移动平均耗时
            history = duration_histories.get(thread_id)
            avg_dur = sum(history) / len(history) if history else 0.0
            
            output_lines.append(
                f"| {thread_id:<20} | {s_count:<8} | {f_count:<8} | "
                f"{last_dur:<15.3f} | {avg_dur:<15.3f} |"
            )
            
        total_success = sum(success_counts.values())
        total_failures = sum(failure_counts.values())

        output_lines.append(separator)
        output_lines.append(
            f"| {'总计':<20} | {total_success:<8} | {total_failures:<8} | "
            f"{'---':<15} | {'---':<15} |"
        )
        output_lines.append("-" * len(separator) + "\n")
        
        tqdm.write("\n".join(output_lines))


    def map_and_collect_results(self, iterable, task_func, result_callback, 
                                input_callback=None, task_args=(), progress_desc="Processing", 
                                monitor_interval=10):
        """
        为可迭代对象中的每一项并发执行任务，并通过回调函数处理结果。
        此方法会阻塞，直到所有提交的任务完成，并定期打印线程状态。

        (参数文档与之前版本相同)
        """
        # 重置所有计数器和跟踪器
        with self._lock:
            self._thread_success_counts.clear()
            self._thread_failure_counts.clear()
            self._thread_last_duration.clear()
            self._thread_duration_history.clear()

        # --- 方法的其余部分与之前版本完全相同 ---
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_item_map = {}
            try:
                total_items = len(iterable)
            except TypeError:
                total_items = None

            with tqdm(total=total_items, desc="已提交任务", dynamic_ncols=True) as pbar:
                for item in iterable:
                    if input_callback and not input_callback(item):
                        continue
                    all_args = (item,) + task_args
                    future = executor.submit(self._task_wrapper, task_func, *all_args)
                    future_to_item_map[future] = item
                    pbar.update(1)

            if not future_to_item_map:
                print("没有任务被提交（可能都被 input_callback 过滤了）。")
                return

            start_time = time.time()
            last_monitor_time = start_time

            with tqdm(total=len(future_to_item_map), desc=progress_desc, unit="task", dynamic_ncols=True) as pbar:
                for future in concurrent.futures.as_completed(future_to_item_map):
                    original_item = future_to_item_map[future]
                    try:
                        result = future.result()
                    except Exception as e:
                        result = e
                        pbar.set_postfix_str(f"最近的错误: {type(result).__name__}")
                    result_callback(result, original_item)
                    pbar.update(1)
                    if monitor_interval and (time.time() - last_monitor_time > monitor_interval):
                        self._print_status(start_time)
                        last_monitor_time = time.time()

        if monitor_interval:
            tqdm.write("\n" + "="*25 + " 所有任务完成 " + "="*25)
            self._print_status(start_time)

