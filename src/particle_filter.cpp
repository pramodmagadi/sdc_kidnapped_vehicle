/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <limits>
#include <cfloat>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    num_particles = 101;
    
    normal_distribution<double> x_init(0, std[0]);
    normal_distribution<double> y_init(0, std[1]);
    normal_distribution<double> theta_init(0, std[2]);
    
    default_random_engine gen;
    
    int i;
    for(i = 0; i < num_particles; i++)
    {
    	Particle particle;
        particle.x = x + x_init(gen);
        particle.y = y + y_init(gen);
        particle.theta = theta + theta_init(gen);
        
        particle.weight = 1;
        particles.push_back(particle);
    }
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	
	default_random_engine gen;
	normal_distribution<double> x_pred(0, std_pos[0]);
	normal_distribution<double> y_pred(0, std_pos[1]);
	normal_distribution<double> theta_pred(0, std_pos[2]);
	for(int i = 0; i < num_particles; i++)
	{
		if(yaw_rate == 0)
		{
		    particles[i].x += velocity * delta_t * cos(particles[i].theta);
		    particles[i].y += velocity * delta_t * sin(particles[i].theta);
		}
		else
		{
	        particles[i].x = particles[i].x + ((velocity/yaw_rate)*(sin(particles[i].theta + (yaw_rate*delta_t)) - sin(particles[i].theta)));
	    
	        particles[i].y = particles[i].y + ((velocity/yaw_rate)*(cos(particles[i].theta) - cos(particles[i].theta + (yaw_rate*delta_t))));
	    
	        particles[i].theta = particles[i].theta + (yaw_rate*delta_t);
	    }
	    particles[i].x += x_pred(gen);
	    particles[i].y += y_pred(gen);
	    particles[i].theta += theta_pred(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for(int i = 0; i < observations.size(); i++)
	{
	    LandmarkObs l = observations[i];
	    
	    int map_id = 0;
	    double min_dist = 10000000.0;
	    
	    for(int j = 0; j < predicted.size(); j++)
	    {
	        double d = dist(predicted[j].x, predicted[j].y, l.x, l.y);
	        if(d < min_dist)
	        {
	            map_id = predicted[j].id;
	            min_dist = d;
	        }
	    }
	    //cout << map_id << endl;
	    observations[i].id = map_id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	for(int i = 0; i < num_particles; i++)
	{
	    //Create a vector of landmarks within sensor range
	    vector<LandmarkObs> predictions;
	    for(int j = 0; j < map_landmarks.landmark_list.size(); j++)
	    {
	        double x_f = map_landmarks.landmark_list[j].x_f;
	        double y_f = map_landmarks.landmark_list[j].y_f;
	        int id_i = map_landmarks.landmark_list[j].id_i;
	        
	        if(dist(x_f, y_f, particles[i].x, particles[i].y) <= sensor_range)
	        {
	            predictions.push_back(LandmarkObs{id_i, x_f, y_f});
	        }
	    }

	    //Transform observations to map coordinate system
	    vector<LandmarkObs> transformed_observations;
	    for(int j = 0; j < observations.size(); j++)
	    {
	        double x_t = observations[j].x*cos(particles[i].theta) - observations[j].y*sin(particles[i].theta) + particles[i].x;
	        double y_t = observations[j].x*sin(particles[i].theta) + observations[j].y*cos(particles[i].theta) + particles[i].y;
	        transformed_observations.push_back(LandmarkObs{observations[j].id, x_t, y_t});
	    }
	    
	    //Associate transformed observations and predictions 
	    dataAssociation(predictions, transformed_observations);
	    for(int k = 0; k < transformed_observations.size(); k++)
	    {
	    cout << "observation = " << transformed_observations[k].id << "x = "<< transformed_observations[k].x << " y = " << transformed_observations[k].y << endl;
	    }
	    //Use multi variate gaussian dist to update weights
	    double std_x = std_landmark[0];
	    double std_y = std_landmark[1];
	    particles[i].weight = 1.0;
	    
	    for(int j = 0; j < transformed_observations.size(); j++)
	    {
	        double pred_x, pred_y;
	        double o_x, o_y;
	        o_x = transformed_observations[j].x;
	        o_y = transformed_observations[j].y;
	        for(int k = 0; k < predictions.size(); k++)
	        {
	            if(predictions[k].id == transformed_observations[j].id)
	            {
	                pred_x = predictions[k].x;
	                pred_y = predictions[k].y;
	            }
	        }
	        double weight_update = (1/(2*M_PI*std_x*std_y)) * exp(-0.5 * ((pow(o_x-pred_x,2)/pow(std_x,2)) + (pow(o_y - pred_y, 2)/pow(std_y,2))));
	        particles[i].weight *= weight_update;
	    }
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    vector<double> weights;
    vector<Particle> resampled_particles;
    default_random_engine gen;
    for(int i = 0; i<num_particles; i++)
    {
        weights.push_back(particles[i].weight);
    }
    
    //get a random starting index
    uniform_int_distribution<int> dist(0,num_particles-1);
    int index = dist(gen);
    
    //maximum weight - here so for loop run time reduces
    double max_weight = *max_element(weights.begin(),weights.end());
    
    double beta = 0.0;
    
    for(int i = 0; i < num_particles; i++)
    {
        beta += (double)rand()/RAND_MAX * 2.0 * max_weight;
        while(beta > weights[index])
        {
            beta -= weights[index];
            index = (index+1)%num_particles;
        }
        resampled_particles.push_back(particles[index]);
    }
    particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
