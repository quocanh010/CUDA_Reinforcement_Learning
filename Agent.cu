



#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <curand.h>
#include <curand_kernel.h>
#include <helper_cuda.h>
#include <helper_functions.h>
//#include "qlearning.h"
//#include "draw_env.h"
//#include "common_def.h"
#include <random>
#include <time.h>


static curandState *states = NULL;
int N = 512;
//int nx_1 = 32;
//int ny_1 = 16;
//dim3 block_1(nx_1, ny_1);
//dim3 grid_1((nx_1 + block_1.x - 1) / block_1.x, (ny_1 + block_1.y - 1) / block_1.y);

// Kernel functions

__global__ void k_Agent_init(curandState *states, float* q_table,  int nx, int ny)
{
	//printf("Hello from init\n");
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int tid = iy * nx + ix;
	//unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	curandState *state = states + 0;
	curand_init(clock() + tid, tid, 0, state);
	if (tid < nx * ny)
	{
		q_table[tid] = curand_uniform(state) * 0.1;
	}


}

__global__ void k_Agent_update_Qtable(short* action, float* Q_table, float gamma, int2* curr_state, int2* next_state, float* reward, int* flag_action)
{
	// printf("Hello from Q update\n");
	//printf("Hello World from update!\n");
	float maxQ;
	float alpha = 0.8;
	int c_index, index;
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int local_flag = flag_action[tid];
	int2 currs = curr_state[tid];
	int2 nexts = next_state[tid];
	float reward_local = reward[tid];
	short local_action = action[tid];
	if (local_flag != 0 && tid < 512)
	{
		//printf("reward = %d\n", local_flag);
		if (reward_local == 0) {
			c_index = (nexts.y * 46 + nexts.x) * 4 + 0;
			maxQ = Q_table[c_index];
			for (int i = 0; i < 4; i++)
			{
				if (Q_table[c_index + i] > maxQ)
				{
					maxQ = Q_table[c_index + i];
				}
			}
		}
		else
		{
			maxQ = 0;
		}
		if(reward_local == 1 || reward_local == -1 || reward_local == 0)
		Q_table[((currs.y) * 46 + currs.x) * 4 + local_action] += alpha * (reward_local + gamma * maxQ - Q_table[((currs.y) * 46 + currs.x) * 4 + local_action]);
		
		if (reward_local != 0)
		{
			flag_action[tid] = 0;
			//printf("reward = %f\n", reward_local);
		}
		//if (reward_local != 1.0 && reward_local != -1.0 && reward_local != 0.0) printf("reward = %f\n", reward_local);
		
	}
	


}



__global__ void k_Agent_adjustepselon(float* k_epsilon)
{

	k_epsilon[0] = k_epsilon[0] - 0.005;
}


__global__ void k_Agent_action(curandState *states, float* Q_table, int2* cstate, short* action, float epselon)
{
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	//__shared__ int2 s[512];
	//float Q = Q_table[tid];
	//float Q_new;
	//unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	//unsigned int iy = threadIdx.x + blockIdx.x * blockDim.x;
	//unsigned int tid = iy * 32 + ix;
	int2 currs = cstate[tid];
	float maxQ = -10000;
	
	curandState *state = states + 0;
	float dOut;

	curand_init(clock() + tid, tid, 0, state);
	dOut = curand_uniform(state);
	
	
	//s[tid] = cstate[tid]; 	//__syncthreads();	
		if (dOut < epselon)
		{

			action[tid] = (short)(curand_uniform(state) * 4);


		}
		else
		{


			//maxQ = Q_table[((currs.y) * 46 + currs.x) * 4];
			//action[tid] = 0;
			for (int i = 0; i < 4; i++)
			{
				if (maxQ < Q_table[((currs.y) * 46 + currs.x) * 4 + i])
				{
					maxQ = Q_table[((currs.y) * 46 + currs.x) * 4 + i];
					action[tid] = (short)i;
				}

			}

		}
	
}

	



__global__ void k_Agent_clearaction(int* flag_agent, short* action)
{
	//printf("Hello from clear\n");
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < 512)
	{
		flag_agent[tid] = 1;
		action[tid] = 0;
	}

	
}




class Agent
{
public:

	float* q_Table;
	float* k_epsilon;
	float epselon = 1.0;
	float gamma = 0.7;
	//int count_alive_agent;
	int *flag_alive;
	short* action;
	int nx = 4;
	int ny = 46 * 46;
	int nBytes = (nx * ny) * sizeof(float);

	Agent();

	~Agent();

	// Mutator
	void Agent_init();
	void Agent_adjustepselon();
	void Agent_update_Qtable(int2* curr_state, int2* next_state, float* reward);
	void Agent_action(int2* cstate);
	void Agent_clearaction();
};


// Create member functions
Agent::Agent()
{

}
Agent::~Agent()
{

}

//Member functions
void Agent::Agent_init()
{
	cudaMalloc((void **)&k_epsilon, 1 * sizeof(float));
	cudaMalloc((void **)&action,  sizeof(short)* N);
	cudaMalloc((void **)&q_Table, nBytes);
	cudaMalloc((void **)&states, sizeof(curandState) * 1 * 1);
	cudaMalloc((void **)&flag_alive, sizeof(int) * N);
	dim3 block(nx, ny);
	dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
	k_Agent_init << <grid, block >> > (states, q_Table, nx, ny);
	k_Agent_clearaction << <8, 64 >> > (flag_alive, action);
}


void Agent::Agent_adjustepselon()
{
	//cudaMemcpy(k_epsilon, &(epselon), sizeof(float), cudaMemcpyHostToDevice);
	//k_Agent_adjustepselon << < 1, 1 >> > (k_epsilon);
	//cudaMemcpy(&epselon, k_epsilon, sizeof(float), cudaMemcpyDeviceToHost);
	epselon = epselon - 0.005;
	if (epselon < 0.0) epselon = 0.0;
}


void Agent::Agent_update_Qtable(int2* curr_state, int2* next_state, float* rewards)
{

	k_Agent_update_Qtable << <8, 64>> > (action, q_Table, gamma, curr_state, next_state, rewards, flag_alive);

}



void Agent::Agent_action(int2* cstate)
{
	
	k_Agent_action << < 8, 64 >> > (states, q_Table, cstate, action, epselon);
}

void Agent::Agent_clearaction()
{
	k_Agent_clearaction << <8, 64 >> > (flag_alive, action);
}


Agent my_agent;


//Interface functions
void agent_init()
{
	my_agent.Agent_init();
}

void agent_clearaction()
{
	my_agent.Agent_clearaction();
}

float agent_adjustepsilon()
{
	my_agent.Agent_adjustepselon();
	return my_agent.epselon;
}

void agent_update(int2* cstate, int2* nstate, float* rewards)
{
	my_agent.Agent_update_Qtable(cstate, nstate, rewards);
}

short* agent_action(int2* cstate)
{

	my_agent.Agent_action(cstate);

	return my_agent.action;
}

