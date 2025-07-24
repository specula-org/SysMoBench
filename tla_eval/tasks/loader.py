"""
Task loader for benchmark test cases.

This module handles loading source code from GitHub repositories
and preparing them as generation tasks with appropriate prompts.
"""

import subprocess
import shutil
from pathlib import Path
from typing import Dict, List
import yaml
from ..methods.base import GenerationTask


class TaskLoader:
    """Loads benchmark tasks by cloning repositories and extracting source code."""
    
    def __init__(self, tasks_dir: str = "tla_eval/tasks", cache_dir: str = "data/repositories"):
        """
        Initialize task loader.
        
        Args:
            tasks_dir: Directory containing task definitions
            cache_dir: Directory to cache cloned repositories
        """
        self.tasks_dir = Path(tasks_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def load_task(self, task_name: str, source_file: str = None) -> GenerationTask:
        """
        Load a specific task by name, automatically cloning repository if needed.
        
        Args:
            task_name: Name of the task (e.g., "etcd")
            source_file: Specific source file path, or None for default
            
        Returns:
            GenerationTask instance with source code and appropriate prompt
        """
        task_dir = self.tasks_dir / task_name
        
        if not task_dir.exists():
            available = self.list_available_tasks()
            raise ValueError(f"Task '{task_name}' not found. Available: {available}")
        
        # Load task metadata
        metadata_file = task_dir / "task.yaml"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Task metadata file not found: {metadata_file}")
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = yaml.safe_load(f)
        
        # Determine which source file to use
        if source_file is None:
            source_file = metadata['default_source_file']
        
        # Find source file info
        source_file_info = None
        for file_info in metadata['source_files']:
            if file_info['path'] == source_file:
                source_file_info = file_info
                break
        
        if source_file_info is None:
            available_files = [f['path'] for f in metadata['source_files']]
            raise ValueError(f"Source file '{source_file}' not found. Available: {available_files}")
        
        # Clone repository and get source code
        source_code = self._get_source_code(metadata['repository'], source_file)
        
        return GenerationTask(
            source_code=source_code,
            task_name=task_name,
            system_type=metadata['system_type'],
            language=metadata['language'],
            description=metadata['description'],
            spec_module=metadata.get('specModule', task_name),  # Use specModule from config or task name as fallback
            # Add additional metadata
            extra_info={
                'file_path': source_file,
                'focus': source_file_info['description'],
                'repository_url': metadata['repository']['url']
            }
        )
    
    def _get_source_code(self, repo_info: Dict, file_path: str) -> str:
        """
        Clone repository if needed and extract source code.
        
        Args:
            repo_info: Repository information from task.yaml
            file_path: Path to source file within repository
            
        Returns:
            Source code content
        """
        repo_url = repo_info['url']
        branch = repo_info.get('branch', 'main')
        
        # Create repository cache directory name from URL
        repo_name = repo_url.split('/')[-1].replace('.git', '')
        repo_cache_dir = self.cache_dir / repo_name
        
        # Clone repository if not already cached
        if not repo_cache_dir.exists():
            print(f"Cloning repository: {repo_url}")
            try:
                subprocess.run([
                    'git', 'clone', '--depth', '1', 
                    '--branch', branch, 
                    repo_url, str(repo_cache_dir)
                ], check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to clone repository {repo_url}: {e.stderr}")
        
        # Read source file
        source_file_path = repo_cache_dir / file_path
        if not source_file_path.exists():
            raise FileNotFoundError(f"Source file not found in repository: {file_path}")
        
        with open(source_file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def get_task_prompt(self, task_name: str, method_name: str) -> str:
        """
        Get the appropriate prompt template for a task and method.
        
        Args:
            task_name: Name of the task
            method_name: Name of the generation method
            
        Returns:
            Prompt template string
            
        Raises:
            FileNotFoundError: If prompt file is not found
        """
        task_dir = self.tasks_dir / task_name
        prompt_file = task_dir / "prompts" / f"{method_name}.txt"
        
        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
        
        with open(prompt_file, 'r', encoding='utf-8') as f:
            return f.read()
    
    def list_available_tasks(self) -> List[str]:
        """List all available task names."""
        if not self.tasks_dir.exists():
            return []
        
        tasks = []
        for item in self.tasks_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                # Check if it has task.yaml
                if (item / "task.yaml").exists():
                    tasks.append(item.name)
        
        return sorted(tasks)
    
    def list_task_source_files(self, task_name: str) -> List[Dict]:
        """List all available source files for a task."""
        task_dir = self.tasks_dir / task_name
        metadata_file = task_dir / "task.yaml"
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = yaml.safe_load(f)
        
        return metadata['source_files']
    
    def get_task_info(self, task_name: str) -> Dict:
        """Get metadata about a specific task."""
        task_dir = self.tasks_dir / task_name
        metadata_file = task_dir / "task.yaml"
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def clear_cache(self, task_name: str = None):
        """
        Clear repository cache.
        
        Args:
            task_name: Clear cache for specific task, or None for all
        """
        if task_name is None:
            # Clear all cache
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            # Clear cache for specific task
            task_info = self.get_task_info(task_name)
            repo_url = task_info['repository']['url']
            repo_name = repo_url.split('/')[-1].replace('.git', '')
            repo_cache_dir = self.cache_dir / repo_name
            
            if repo_cache_dir.exists():
                shutil.rmtree(repo_cache_dir)


# Global task loader instance
_task_loader = None

def get_task_loader() -> TaskLoader:
    """Get global task loader instance."""
    global _task_loader
    if _task_loader is None:
        _task_loader = TaskLoader()
    return _task_loader

def load_task(task_name: str, source_file: str = None) -> GenerationTask:
    """Convenience function to load a task."""
    return get_task_loader().load_task(task_name, source_file)