
def get_exclude_list(log_file):
    exclude_list = list()
    with open(log_file, 'r', newline='') as log:
        for line in log:
            line = line.rstrip().split('_corr')[0]
            if line == 'dataset_name':
                continue
            exclude_list.append(line+'.csv')
    return exclude_list
