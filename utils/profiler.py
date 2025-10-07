"""
성능 측정 프로파일링 도구
=========================
Author: Jin
Date: 2025.09.17
Version: 1.0

Description:
함수의 실행 시간과 메모리 사용량을 측정하는 데코레이터 기반 프로파일링 도구입니다.
timeit과 sys.getsizeof를 활용하여 정확한 성능 측정을 제공하며,
시간 측정, 메모리 측정, 종합 성능 측정의 세 가지 데코레이터를 제공합니다.
"""
import timeit
import sys
import functools
import asyncio
from typing import Callable, List
from config.logging_config import logger

def measure_time(number: int = 1000, repeat: int = 3) -> Callable:
    """
    함수 실행 시간을 측정하는 데코레이터
    
    Args:
        number: timeit 실행 횟수 
        repeat: 반복 측정 횟수
    
    Usage:
        @measure_time()
        def my_function():
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 여러 번 측정해서 평균값 사용
            times: List[float] = timeit.repeat(lambda: func(*args, **kwargs), number=number, repeat=repeat)
            min_time: float = min(times)
            avg_time: float = sum(times) / len(times)
            
            # 1회 실행 시간 계산
            min_per_call: float = min_time / number
            avg_per_call: float = avg_time / number
            
            logger.info(f"execution time of [{func.__name__}]:")
            logger.info(f" - minimum: {min_per_call*1000:.4f}ms")
            logger.info(f" - average: {avg_per_call*1000:.4f}ms")
        
            return func(*args, **kwargs) 
        return wrapper
    return decorator


def measure_memory() -> Callable:
    """
    함수 실행 전후 메모리 사용량을 측정하는 데코레이터
    
    Usage:
        @measure_memory()
        def my_function():
            data = [i for i in range(10000)]
            return data
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 함수 실행 전 메모리 측정
            logger.info(f"[{func.__name__}] 메모리 사용량 측정 시작")
                
            # 인자들의 메모리 사용량
            args_memory: int = sum(sys.getsizeof(arg) for arg in args)
            kwargs_memory: int = sum(sys.getsizeof(k) + sys.getsizeof(v) for k, v in kwargs.items())
            
            logger.info(f"- 입력 인자 메모리: {args_memory + kwargs_memory:,} bytes")
            
            result = func(*args, **kwargs)
            
            # 결과값의 메모리 사용량 측정
            result_memory: int = sys.getsizeof(result)
            logger.info(f"[{func.__name__}] 반환값 메모리: {result_memory:,} bytes ({result_memory/1024:.2f} KB)")
            
            return result
        
        return wrapper
    return decorator


def measure_performance(
    time_number: int = 1000, 
    time_repeat: int = 3, 
) -> Callable:
    """
    시간과 메모리를 동시에 측정하는 종합 성능 데코레이터
    
    Args:
        time_number: timeit 실행 횟수
        time_repeat: 시간 측정 반복 횟수

    Usage:
        @measure_performance()
        def my_function():
            # 코드
            pass
    """
    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):
            # async 함수 처리
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                import time
                logger.info(f"[{func.__name__}] 성능 측정 시작 (async)")
                
                start_time = time.time()
                result = await func(*args, **kwargs)
                end_time = time.time()
                
                execution_time = (end_time - start_time) * 1000  # ms
                result_memory: int = sys.getsizeof(result)
                
                logger.info(f"실행 시간: {execution_time:.4f}ms")
                logger.info(f"메모리 사용량: {result_memory:,} bytes ({result_memory/1024:.2f} KB)")
                logger.info(f"[{func.__name__}] 성능 측정 완료\n")
                
                return result
            return async_wrapper
        else:
            # 일반 함수 처리
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                logger.info(f"[{func.__name__}] 종합 성능 측정 시작")
                
                times: List[float] = timeit.repeat(lambda: func(*args, **kwargs), number=time_number, repeat=time_repeat)
                
                min_time: float = min(times)
                avg_time: float = sum(times) / len(times)
                
                min_per_call: float = min_time / time_number
                avg_per_call: float = avg_time / time_number
                
                # 실제 실행 및 메모리 측정
                result = func(*args, **kwargs)
                result_memory: int = sys.getsizeof(result)
                
                # 결과 출력
                logger.info(f"실행 시간:")
                logger.info(f"- 최소: {min_per_call*1000:.4f}ms")
                logger.info(f"- 평균: {avg_per_call*1000:.4f}ms")
                logger.info(f"- 측정: {time_number}회 × {time_repeat}번")
                
                logger.info(f"메모리 사용량:")
                logger.info(f"- 반환값: {result_memory:,} bytes ({result_memory/1024:.2f} KB)")
                 
               
                logger.info(f"[{func.__name__}] 성능 측정 완료\n")
                
                return result
            return wrapper
    return decorator
