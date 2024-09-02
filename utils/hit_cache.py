from multiprocessing import Process
import shutil
import uuid
import multiprocessing
import os
import fcntl
import json
import stat
import random
import time
import warnings


def __read_file(p):
    with open(p, 'rb') as f:
       _ = f.read()
    print('Loaded:', p)


def hit_cache(path):
    warnings.warn("hit_cache may be useless, please use RamDiskCache ...")
    p = multiprocessing.Process(target=__read_file, args=(path,))
    p.start()


def lock(f):
    fcntl.flock(f.fileno(), fcntl.LOCK_EX)

 
def unlock(f):
    fcntl.fcntl(f.fileno(), fcntl.LOCK_UN)


class RamDiskCache:
    cfg_name='usage.json'

    def __init__(self, root, capacity, warning_mem=3, cfg_name='usage.json') -> None:
        """
        root: the cache root
        capacity: the size of cache file, GB
        warning_mem: stop caching if the left memory is smaller than warning_mem (GB)
        """
        self.root = root
        self.capacity = capacity*1024   #MB
        self.warning_mem = warning_mem*1024   #MB
        self.used_mem = 0
        self.maintain = {}   #{src: (target, status)}, status True, False, True: caching finished, False: cacheing 
        self.cfg_name = cfg_name
        self.init()

    def init(self):
        p = os.path.join(self.root, self.cfg_name)
        print('Creating config file at:', p)
        if not os.path.exists(p):
            self.__write_cfg()
        else:
            self.__read_cfg()
            self.__clean()
            self.__write_cfg()
            self.used_mem = self.__cal_used_mem()
    
    def __clean(self):
        temp = {}
        for src, (target, status) in self.maintain.items():
            if os.path.exists(target):
                temp[src] = (target, status)
            else:
                print('{} is not exists, updating cfg file'.format(target), flush=True)
        # remove files not exist in the maintain list
        all_files = [os.path.join(self.root, f) for f in os.listdir(self.root) if f!='use.json']
        all_files = [i for i in all_files if os.path.isfile(i)]
        # for f in all_files:
        #     if f not in temp.values():
        #         os.remove(f)
        #         print('removing:', f)
        self.maintain = temp
    
    def add_cache(self, path, join=False):
        self.__read_cfg()
        left_mem = self.capacity - self.used_mem
        while left_mem < self.warning_mem:
            src = random.choice(self.maintain.keys())
            print('Warning, Left space: {} MB, removing {}'.format(left_mem, self.maintain[src]))
            self.remove_cache(src)
            self.used_mem = self.__cal_used_mem()
            left_mem = self.capacity - self.used_mem

        if isinstance(path, str):
            path = [path]
        for p in path:
            if p in self.maintain.keys():
                target_path, status = self.maintain[p]
                if status and os.path.exists(target_path):
                    print('{} exists, corresponding file is: {}'.format(p, target_path))
                elif status:
                    raise RuntimeError('{} is in the maintain list, but {} does not exist'.format(p, target_path))
                else:
                    print('Other process is caching this file, skip ...')
            else:
                process = Process(target=self.__copy_file, args=(p,))
                process.start()
                if join:
                    process.join()
    
    def remove_cache(self, path):
        self.__read_cfg()

        if isinstance(path, str):
            path = [path]
        path = [p for p in path if p in self.maintain.keys()]
        for p in path:
            process = Process(target=self.__remove_file, args=(p, ))
            process.start()

    def new_path(self, src):
        self.__read_cfg()
        if src in self.maintain.keys():
            target, status = self.maintain[src]
            if not status:
                target = src
        else:
            target = src
        return target
    
    def re_copy(self, src):
        self.__read_cfg()
        if src in self.maintain.keys():
            target, status = self.maintain[src]
            if status and not os.path.exists(target):
                return True
        return False
            

    def __copy_file(self, src):
        # generate temp file name
        postfix = src.split('.')[-1]
        name = str(uuid.uuid3(namespace=uuid.NAMESPACE_DNS, name=src)) + '.' + postfix
        target = os.path.join(self.root, name)

        s = time.time()
        self.increase_cfg(src, (target, False))
        shutil.copyfile(src=src, dst=target, follow_symlinks=False)
        self.increase_cfg(src, (target, True), finished=True)
        print('Cache time:', time.time() - s, flush=True)
    
    def __remove_file(self, src):
        self.__read_cfg()
        target, status = self.maintain[src]
        if os.path.exists(target) and status:
            os.remove(target)
            self.decrease_cfg(src)
            print('remove successfuly:', target)
        else:
            print('Removing failed, {} does not exist...'.format(target))
    
    def __cal_used_mem(self):
        used_mem = 0
        self.__read_cfg()
        for _, (v, status) in self.maintain.items():
            if status:
                if not os.path.exists(v):
                    print('Warning, cache file inconsistency.', v, ': does not exist ...')
                else:
                    size = int(os.path.getsize(v)/1024/1024) #MB
                    used_mem += size
            else:
                print('{} is caching, skip calculating memory for this file'.format(v))

        return used_mem
    
    def __read_cfg(self) -> dict:
        p = os.path.join(self.root, self.cfg_name)
        counter = 1
        while True:
            try:
                with open(p) as f:
                    lock(f)
                    self.maintain = json.load(f)
                    unlock(f)
            except:
                print('Failed to load json ...')
                counter += 1
            else:
                print('Sucessfully loaded json file ...')
                break
            if counter > 5:
                break           

    def __write_cfg(self):
        p = os.path.join(self.root, self.cfg_name)
        with open(p, 'w') as f:
            lock(f)
            json.dump(self.maintain, f)
            f.flush()
            unlock(f)
            
        rights = oct(os.stat(p).st_mode)[-3:]
        if rights != '777':
            os.chmod(p, mode=stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
            rights = oct(os.stat(p).st_mode)[-3:]
            print('{} mode is: {}'.format(p, rights))

    def increase_cfg(self, src, target, finished=False):
        # load cfg, other process may update it 
        self.__read_cfg()
        # update maintain list
        self.maintain[src] = target
        # write back the maintain list 
        self.__write_cfg()
        # update memory 
        if finished:
            self.used_mem = self.__cal_used_mem()
            print('Sucessfully cached: {}, used storage: {:.2f} GB, left space: {:.2f} GB'.format(
                src, self.used_mem/1024, (self.capacity - self.used_mem)/1024
            ), flush=True)

    def decrease_cfg(self, src):
        # load cfg, other process may update it
        self.__read_cfg()
        # update maintain list
        if src in self.maintain.keys():
            _ = self.maintain.pop(src)
        else:
            print('{} does not exist in the maintain list ...'.format(src))
        # write back the maintain list 
        self.__write_cfg()
        # update memory 
        self.used_mem = self.__cal_used_mem()
        print('Sucessfully removed: {}, used storage: {:.2f} GB, left space: {:.2f} GB'.format(
            src, self.used_mem/1024, (self.capacity - self.used_mem)/1024
        ))


if __name__ == '__main__':
    import time
    # step 1, create ramdisk use following cmd, ask help from your admin. Of course, you can also use a SSD path as buffer
    # sudo mount -t tmpfs -o rw,size=100G tmpfs /mnt/ramdisk
    # Step 2, use
    def fake_work(p):
        time.sleep(2)
        print(p, ':is finished...')

    root = '/home/jmabq/temp'
    paths = [os.path.join(root, i) for i in os.listdir(root)]

    cache_engine = RamDiskCache('/mnt/ramdisk', 100)

    for index in range(len(paths)):

        curr_path = paths[index]
        print('current:', curr_path)
        next_index = (index + 1) % len(paths)
        next_path = paths[next_index]

        # background cache
        cache_engine.add_cache(next_path)   # if you want to cache more, pass it a list such as [path1, path2]

        # find new path, if the path is alread cached, then it will return the cached path, else return the input
        cache_path = cache_engine.new_path(curr_path)
        # current work
        fake_work(cache_path)
        # remove cached file
        cache_engine.remove_cache(curr_path) # manulary remove cached file, if you forget this, I will randomly pop used items if the memory is not enough.
