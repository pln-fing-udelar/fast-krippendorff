#!/usr/bin/env python
import krippendorff
import numpy as np


def transpose_list(array):
    return list(map(list, zip(*array)))


def main():
    print("Example from http://en.wikipedia.org/wiki/Krippendorff's_Alpha")
    print('')

    data = (
        "*    *    *    *    *    3    4    1    2    1    1    3    3    *    3",  # coder A
        "1    *    2    1    3    3    4    3    *    *    *    *    *    *    *",  # coder B
        "*    *    2    1    3    4    4    *    2    1    1    3    3    *    4",  # coder C
    )
    print('\n'.join(data))
    print('')

    unit_counts = np.array([[sum(1 for r in unit if r == '1'),
                             sum(1 for r in unit if r == '2'),
                             sum(1 for r in unit if r == '3'),
                             sum(1 for r in unit if r == '4')]
                            for unit in transpose_list([coder.split() for coder in data])])
    print("Krippendorff alpha for nominal metric: ", krippendorff.alpha(unit_counts, level_of_measurement='nominal'))
    print("Krippendorff alpha for interval metric: ", krippendorff.alpha(unit_counts))


if __name__ == '__main__':
    main()
