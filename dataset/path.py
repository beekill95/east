import os


def does_directory_exist(dir):
    return os.path.isdir(dir)


def does_file_exist(file):
    return os.path.isfile(file)


def list_all_files(dir):
    if not does_directory_exist(dir):
        raise NotADirectoryError(f'{dir} does not exists.')

    return os.listdir(dir)


def list_all_files_with_exts(dir, exts, ignore_case=True):
    """
    List all files in |dir| end with extensions |exts|.

    :param dir: path to directory.
    :param exts: a list of extensions of files.
    :param ignore_case: whether to check extensions case sensitive or not.
    :return: files in |dir| with the extensions.
    """
    files = list_all_files(dir)

    exts = [e.lower() if ignore_case else e for e in exts]
    files_with_exts = []
    for f in files:
        splitted = f.split('.')
        ext = splitted[-1].lower() if ignore_case else splitted[-1]
        if len(splitted) > 1 and ext in exts:
            files_with_exts.append(f)

    return files_with_exts


def list_all_images(dir):
    img_exts = ['png', 'jpeg', 'jpg']
    return list_all_files_with_exts(dir, img_exts)


def get_file_name(path, with_ext=True):
    file_name = path.split('/')[-1]
    return file_name if with_ext else '.'.join(file_name.split('.')[:-1])


def join_path(path, *paths):
    return os.path.join(path, *paths)


def make_dirs(dir):
    if not does_directory_exist(dir):
        os.makedirs(dir)


def absolute_path(path):
    return os.path.abspath(path)


def symlink(link, target):
    os.symlink(target, link)
