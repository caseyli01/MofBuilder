from pathlib import Path


def get_data_path():
    """
    Returns location of data files within module.

    :return:
        The location of data files within module.
    """
    #multiple levels up to reach the root directory
    return Path(__file__).parents[3] / "database"

if __name__ == "__main__":
    print(get_data_path())
