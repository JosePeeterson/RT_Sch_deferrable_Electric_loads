Two papers are implemented as part of a graded assessment in the Module, EE5903 Real-Time Systems during my Masters program in NUS.

# Paper 1: Real-time load balancing scheduling algorithm for periodic
simulation models
## Abstract
In this simulation we develop a scheduling table for scheduling independent and periodic tasks. Scheduling of a static and dynamic model set in a time-stepped simulation is carried out using the time-stepped load balancing scheduling (TLS) algorithm. The algorithm ensures correct order of execution so that models meet deadlines. It is also able to balance the load during model addition or deletion to spread out tasks evenly across sub steps of periods and to balance the computation times and improve real-time reliability.
## Short Introduction of the Problem, model, and objective(s)
In Combat simulations, an entity like a tank will have models such as radar sensors that are periodically scanning for enemies. As models maybe destroyed or added during combat, we combine offline scheduling and online adaptive methods to create a task schedule table. Tasks to run at each time step is decided by the TLS algorithm and updated in the scheduling table at run-time. Conventional algorithms like EDF calculate and sort deadlines online and this incurs a high overhead. Sometimes when models are created, or existing models are deleted, priority changes in EDF can occur and task computation times may exceed beyond its current period causing jitter. Jitter count, which is the variability in starting time or completion time of the same periodic task at different periods is higher with EDF when the number of model sets increases. A better scheduler is characterized by a smaller jitter count and fewer resource requirements to handle it. The issues of missing deadlines and infeasible schedules is tackled by improving real-time reliability.

Figure 1: Initial static schedule table
![image](https://github.com/user-attachments/assets/1be70091-c482-4362-b9cd-4c567b686df6)

Figure 2:Dynamic schedule table after model addition
![image](https://github.com/user-attachments/assets/60662d10-479a-497f-b605-3464ee2fddf7)


![image](https://github.com/user-attachments/assets/198f83db-e8ff-48d1-97be-3eae943b2c29)


# Paper 2: Real-time Scheduling of Deferrable Electric Loads
## Abstract
In this real-time Electric vehicle charging schedule task, we analyse ways to reduce grid energy use and maximize renewable energy use to deliver energy over a servicing period to deferrable loads using real-time heuristic causal scheduling policies like EDF and Receding Horizon Control (RHC). RHC utilizes more renewable energy than EDF.

## Short Introduction of the Problem, model, and objective(s)
Certain electric loads like electric vehicles are flexible in the sense that given the energy required to fully charge and maximum charging power we can schedule this process/task over some time. This flexibility allows us to incorporate variable energy sources like wind energy using power forecast data together with readily available grid energy sources.
Causal optimal scheduling policies do not exist where there are power constraints. This is because future power availability has an impact on power distribution to loads at present time. This is illustrated in theorem 1 in [2]. Currently, the grid energy is reserved in high quantities to counter renewable energy variability, but this is not a scalable solution. Therefore, heuristic causal scheduling policies through EDF and RHC algorithms are proposed and their performance evaluated using metrics of required renewable and grid energy.

<img width="564" alt="image" src="https://github.com/user-attachments/assets/eecd25a2-dfa8-40b2-b82f-909f58a2a3be">

<img width="512" alt="image" src="https://github.com/user-attachments/assets/857fac80-b91c-4f36-821d-712420bb7009">


<img width="341" alt="image" src="https://github.com/user-attachments/assets/5a673ff2-33b9-4ca4-976b-393d76680465">
