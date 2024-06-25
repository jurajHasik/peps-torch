def _debug_allocated_tensors(device=None,global_args=None,totals_only=False):
    if global_args and not device:
        cuda= global_args.device
        if cuda == "cpu":
            cuda= global_args.offload_to_gpu
            if cuda in ['None','none','NONE']:
                return
    if device:
        cuda= device
    if cuda and cuda!=torch.device("cpu"):
        torch.cuda.synchronize(device=cuda)
    report=""
    try:
        af.device.sync(device=af.device.get_device())
        _af_mem= af.device.device_mem_info()
        _af_mem['alloc']['bytes_GiB']= _af_mem['alloc']['bytes']/1024**3
        _af_mem['lock']['bytes_GiB']= _af_mem['lock']['bytes']/1024**3
        report=report+f"AF {_af_mem}\n"
    except:
        pass
    tot_cuda=0
    _t= {}
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                if obj.data_ptr() in _t:
                    _t[obj.data_ptr()]['count']=+1
                else:
                    _t[obj.data_ptr()]={'count': 1, 'device': {obj.device}, 'shape': obj.size(), \
                        'size': obj.numel()*obj.element_size()}
        except: 
            pass
    if not totals_only:
        for ptr,row in _t.items():
            report=report+f"{ptr} {row['count']} {row['device']} {row['shape']}\n"
            tot_cuda+= row['size']
    report=report+f"tot_cuda {tot_cuda/1024**3} GiB\n"
    if cuda and cuda!="cpu":
        try:
            cp=subprocess.run(["nvidia-smi"], capture_output=True, text=True)
            report=report+cp.stdout
        except:
            pass
        a,t= torch.cuda.mem_get_info()
        report=report+f"alloc/reserved {a/1024**3} GiB total {t/1024**3} GiB\n"
        report=report+f"alloc {torch.cuda.memory_allocated()/1024**3} GiB\n"
        report=report+f"reserved {torch.cuda.memory_reserved()/1024**3} GiB\n"
    return report