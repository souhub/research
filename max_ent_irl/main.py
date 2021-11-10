from models import frozen_lake_v0, logistic_curve
import time


def main():
    start = time.time()
    logistic_curve()
    done = time.time()
    elapsed_time = done-start
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")


main()
