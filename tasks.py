tasks_monusac = {
    # 0: Background
    # 1: Epithelial
    # 2: Lymphocyte
    # 3: Macrophage
    # 4: Neutrophil
    "offline": {
        0: [0, 1, 2, 3, 4],
    },
    "3-1": {
        0: [0, 1, 2, 3],
        1: [4],
    },
    "2-2": {
        0: [0, 1, 2],
        1: [3, 4],
    },
    "1-1": {0: [0, 1], 1: [2], 2: [3], 3: [4]},
}
tasks_consep = {
    "offline": {
        0: [0, 1, 2, 3],
    },
    "2-1": {
        0: [0, 1, 2],
        1: [3],
    },
    "1-1": {0: [0, 1], 1: [2], 2: [3]},
}


def get_task_list():
    return list(tasks_monusac.keys()) + list(tasks_consep.keys())


def get_task_labels(dataset, name, step):
    if dataset == "monusac":
        task_dict = tasks_monusac[name]
    elif dataset == "consep":
        task_dict = tasks_consep[name]
    else:
        raise NotImplementedError
    assert (
        step in task_dict.keys()
    ), f"You should provide a valid step! [{step} is out of range]"

    labels = list(task_dict[step])
    labels_old = [label for s in range(step) for label in task_dict[s]]
    return labels, labels_old, f"data/{dataset}/{name}"


def get_per_task_classes(dataset, name, step):
    if dataset == "monusac":
        task_dict = tasks_monusac[name]
    elif dataset == "consep":
        task_dict = tasks_consep[name]
    else:
        raise NotImplementedError
    assert (
        step in task_dict.keys()
    ), f"You should provide a valid step! [{step} is out of range]"

    classes = [len(task_dict[s]) for s in range(step + 1)]
    return classes
