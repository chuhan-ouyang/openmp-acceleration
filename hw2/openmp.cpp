#include "common.h"
#include <cmath>
#include <vector>
#include <iostream>
#include <omp.h>

using namespace std;

// define bins as an array of vector<particle_t*>
typedef vector<particle_t*> bin_t;
bin_t* bins;
int numRows;
int totalBins;
omp_lock_t* lckArray;
vector<vector<int>> dirs;

// Apply the force from neighbor to particle
void apply_force(particle_t& particle, particle_t& neighbor) {
    // Calculate Distance
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    // Check if the two particles should interact
    if (r2 > cutoff * cutoff)
        return;

    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);

    // Very simple short-range repulsive force
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

// Integrate the ODE
void move(particle_t& p, double size) {
    // Slightly simplified Velocity Verlet integration
    // Conserves energy better than explicit Euler method
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x += p.vx * dt;
    p.y += p.vy * dt;

    // Bounce from walls
    while (p.x < 0 || p.x > size) {
        p.x = p.x < 0 ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }

    while (p.y < 0 || p.y > size) {
        p.y = p.y < 0 ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }
}


void init_simulation(particle_t* parts, int num_parts, double size) {
	// You can use this space to initialize static, global data objects
    // that you may need. This function will be called once before the
    // algorithm begins. Do not do any particle simulation here
    numRows = (size / cutoff) + 1;
    totalBins = numRows * numRows;
    bins = new bin_t[totalBins];
    lckArray = new omp_lock_t[totalBins];

    for (int i = 0; i < totalBins; i++)
    {
        omp_init_lock(&lckArray[i]);
    }

    dirs = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 0}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {

    #pragma omp for
    for (int i = 0; i < totalBins; ++i) {
        bins[i].clear();
    }

    #pragma omp for
    for (int i = 0; i < num_parts; ++i) 
    {
        int col = parts[i].x / cutoff;
        int row = parts[i].y / cutoff;
        int bin = col + row * numRows;
        parts[i].ax = parts[i].ay = 0;
        omp_set_lock(&(lckArray[bin]));
        bins[bin].push_back(&parts[i]);
        omp_unset_lock(&(lckArray[bin]));
    }
 
    #pragma omp for collapse(3)
    for (int r = 0; r < numRows; ++r)
    {
        for (int c = 0; c < numRows; ++c)
        {
            for (int j = 0; j < dirs.size(); j++)
            {
                if (0 <= (r + dirs[j][0]) && (r + dirs[j][0]) < numRows && 0 <= (c + dirs[j][1]) && (c + dirs[j][1]) < numRows)
                {   
                    for (int i = 0; i < bins[(r * numRows + c)].size(); ++i)
                    {
                        for (int k = 0; k < bins[(r + dirs[j][0]) * numRows + (c + dirs[j][1])].size(); ++k)
                        {
                            apply_force(*bins[r * numRows + c][i], *bins[(r + dirs[j][0]) * numRows + (c + dirs[j][1])][k]);
                        }
                    }
                }
            }
        }
    }

    #pragma omp for
    // Move Particles
    for (int i = 0; i < num_parts; ++i) {
        move(parts[i], size);
    }
}