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
omp_lock_t write_lock;

void apply_force_pairs(particle_t& particle, particle_t& neighbor) {
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
    omp_set_lock(&(write_lock));
    particle.ax += coef * dx;
    particle.ay += coef * dy;
    neighbor.ax += coef * (-dx);
    neighbor.ay += coef * (-dy);
    omp_unset_lock(&(write_lock));
}

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
    omp_init_lock(&write_lock);
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
        parts[i].ax = parts[i].ay = 0; 
        int bin = col + row * numRows;
        omp_set_lock(&(lckArray[bin]));
        bins[bin].push_back(&parts[i]);
        omp_unset_lock(&(lckArray[bin]));
    }

    #pragma omp for
    for (int i = 0; i < totalBins; i++)
    {
        for (int j = 0; j < bins[i].size(); j++)
        {
            for (int k = j + 1; k < bins[i].size(); k++)
            {
                apply_force_pairs(*bins[i][j], *bins[i][k]);
            }
        }
    }

    #pragma omp for
    // for each particle, only apply force onto it for particles in the 9 neighboring bins
    for (int i = 0; i < num_parts; ++i)
    {

        int col = parts[i].x / cutoff;  
        int row = parts[i].y / cutoff; 
        int binNum = col + row * numRows;

        bool hasLeft = col - 1 >= 0;
        bool hasRight = col + 1 < numRows;
        bool hasTop = row - 1 >= 0;
        bool hasBottom = row + 1 < numRows;

        // current bin
        if (hasRight)
        {
            for (int j = 0; j < bins[binNum + 1].size(); ++j)
            {
                apply_force_pairs(parts[i], *bins[binNum + 1][j]);
            }
        }

       if (hasBottom)
        {
            //cout << "7" << endl;
            // bin directly below
            for (int j = 0; j < bins[binNum + numRows].size(); ++j)
            {
                apply_force_pairs(parts[i], *bins[binNum + numRows][j]);
            }

            //cout << "8" << endl;
            if (hasLeft)
            {
                for (int j = 0; j < bins[binNum + numRows - 1].size(); ++j)
                {
                    apply_force_pairs(parts[i], *bins[binNum + numRows - 1][j]);
                }
            }
            
            // bin directly below to the left
            //cout << "9" << endl;
            if (hasRight)
            {
                for (int j = 0; j < bins[binNum + numRows + 1].size(); ++j)
                {
                    apply_force_pairs(parts[i], *bins[binNum + numRows + 1][j]);
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