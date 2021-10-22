from experiments import *

if __name__ == '__main__':
    experiments = []
    print("Hello and welcome to my sub optimal implementation of AlphaZero.")
    print("Made by Thomas Volden <thvo@itu.dk>")
    print()
    
    print("Please choose an experiment to run:")
    for i, e in enumerate(experiment_list):
        print(f" {i}) {e[0]}")
    
    print()
    ex = input(f"Choose experiment [0-{len(experiment_list)-1}]: ")
    while not ex.isnumeric() and int(ex) not in range(len(experiment_list)):
        print("Please write the number of the experiment you want to run.")
        ex = input(f"Choose experiment [0-{len(experiment_list)-1}]: ")
    
    print("Running experiment:")
    print(experiment_list[int(ex)][0])

    experiment_list[int(ex)][1]()