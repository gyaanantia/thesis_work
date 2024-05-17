def get_experiment_name(var, qual=None):
    name = "sd"
    with open("/home/gyaan/data/sdzoo/onpolicy/scripts/train_sd_scripts/current_change.txt", "r") as change_file:
        change = change_file.readline().strip()
        
    name += f"-{var}"
    if qual is not None:
        name += f"-{qual}"
    name += f"-{change}"
    return name
