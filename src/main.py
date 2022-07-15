import mlflow
import argparse


def main():
    with mlflow.start_run() as run:
        mlflow.run('.', 'stage01', use_conda=False)
        mlflow.run('.', 'stage02', use_conda=False)
        mlflow.run('.', 'stage03', use_conda=False)
        mlflow.run('.', 'stage04', use_conda=False)

if __name__=='__main__':
    main()