#include "common.h"
#include <cmath>
#include <unordered_map>
#include <vector>
#include <map>

using namespace std;

// global hashmap: mapping a particle to a row/col consudensed array
map<vector<int>, vector<particle_t>> bins;
vector<vector<int>> dirs;

// Apply the force from neighbor to particle
void apply_force(particle_t& particle, particle_t& neighbor) {
    // apply force on each other?
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

    // for all particles, compute their row and col bin, add to hashmap
    for (int i = 0; i < num_parts; i++)
    {
        particle_t p = parts[i];
        int row = p.x / cutoff;
        int col = p.y / cutoff;
        bins[{row, col}].push_back(p);
    }
    
    dirs = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 0}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // Compute Forces

    // for each particle, get the row and column
    // for all 9 in dirs, compute the new row and col, and then look through all in that bin
    for (int i = 0; i < num_parts; i++)
    {
        particle_t p = parts[i];
        int row = p.x / cutoff;
        int col = p.y / cutoff;
        bins[{row, col}].push_back(p);
    }

    for (int i = 0; i < num_parts; ++i) {
        parts[i].ax = parts[i].ay = 0;
        for (int j = 0; j < num_parts; ++j) {
            apply_force(parts[i], parts[j]);
        }
    }


    for (int i = 0; i < num_parts; i++)
    {
        vector<particle_t> neighbors;
        particle_t p = parts[i];
        int row = p.x / cutoff;
        int col = p.y / cutoff;
        parts[i].ax = parts[i].ay = 0;
        for (int j = 0; j < dirs.size(); j++)
        {
            int new_row = row + dirs[j][0];
            int new_col = col + dirs[j][1];
            for (int k = 0; k < bins[{new_row, new_col}].size(); k++)
            {
                neighbors.push_back(bins[{new_row, new_col}][k]);
            }
        }
        for (int j = 0; j < neighbors.size(); j++)
        {
            apply_force(parts[i], neighbors[j]);
        }
    }

    for (int i = 0; i < num_parts; ++i) {
        move(parts[i], size);
    }
}
