/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;
using std::default_random_engine;
void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 101;  // TODO: Set the number of particles
  default_random_engine gen;
  //creating normal distribution for x,y,theta and uncertainties
  normal_distribution<double> dist_x(x,std[0]);
  normal_distribution<double> dist_y(y,std[1]);
  normal_distribution<double> dist_theta(theta,std[3]);
  
  for(unsigned int i= 0; i< num_particles; i++){
    Particle iparticle;
    iparticle.weight = 1.0;
    iparticle.x = dist_x(gen);
    iparticle.y = dist_y(gen);
    iparticle.theta =dist_theta(gen);
    
    particles.push_back(iparticle);
    weights.push_back(iparticle.weight);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
   default_random_engine gen;
   for(int i=0; i<num_particles;i++){
     //check if yaw rate is approximately zero or not
     if(fabs(yaw_rate) < 0.0001){
       particles[i].x = particles[i].x + velocity*delta_t*cos(particles[i].theta);
       particles[i].y = particles[i].y + velocity*delta_t*sin(particles[i].theta);
       particles[i].theta = particles[i].theta;
     }
     else{
       particles[i].x = particles[i].x + ((velocity/yaw_rate)*(sin(particles[i].theta + (yaw_rate*delta_t)) - sin(particles[i].theta)));
       particles[i].y = particles[i].y + ((velocity/yaw_rate)*(-cos(particles[i].theta + (yaw_rate*delta_t)) + cos(particles[i].theta)));
       particles[i].theta = particles[i].theta + yaw_rate*delta_t;
     }
     normal_distribution<double> dist_x(particles[i].x, std_pos[0]);
     normal_distribution<double> dist_y(particles[i].y, std_pos[1]);
     normal_distribution<double> dist_theta(particles[i].theta, std_pos[3]);
     
     particles[i].x = dist_x(gen);
     particles[i].y = dist_y(gen);
     particles[i].theta = dist_theta(gen);                                   
   }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
    for(unsigned int i = 0 ; i < observations.size(); i++){
      double closest_dist = std::numeric_limits<double>::max();
      int closest_mapid = -1;
      
      for(unsigned int j =0 ; j < predicted.size(); j++){
        double distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
        if(distance < closest_dist){
          closest_dist = distance;
          closest_mapid = predicted[j].id;
        }
      }
      observations[i].id = closest_mapid;
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  double curr_x, curr_y, curr_theta, weight_p = 0.0;
  float mp_x, mp_y;
  int mi;
  
  for(unsigned int i = 0; i < num_particles; i++){
   vector<LandmarkObs> TObs;
    curr_x = particles[i].x;
    curr_y = particles[i].y;
    curr_theta = particles[i].theta;
    
    vector<LandmarkObs> closestpred;
    
    for(unsigned int j =0; j < map_landmarks.landmark_list.size() ; j++){
      
      mp_x = map_landmarks.landmark_list[j].x_f;
      mp_y = map_landmarks.landmark_list[j].y_f;
      mi = map_landmarks.landmark_list[j].id_i;
      
      if(fabs(mp_x - curr_x) <= sensor_range && fabs(mp_y - curr_y) <=sensor_range){
          closestpred.push_back(LandmarkObs{mi, mp_x, mp_y});
      }
    }
    double tobs_x;
    double tobs_y;
    for(unsigned int j =0 ;j < observations.size(); j++){
      tobs_x = (cos(curr_theta)*observations[j].x) - (sin(curr_theta)*observations[j].y) + curr_x;
      tobs_y = (sin(curr_theta)*observations[j].x) + (cos(curr_theta)*observations[j].y) + curr_y;
      int tobs_id = observations[j].id;
      TObs.push_back(LandmarkObs{tobs_id, tobs_x, tobs_y});
    }
    
    dataAssociation(closestpred,TObs);
    
    particles[i].weight = 1.0;
    double sig_x, sig_y, x_obs, y_obs, mu_x, mu_y, gauss_norm, exponent;
    sig_x = std_landmark[0];
    sig_y = std_landmark[1];
    
    for(unsigned int j =0; j < TObs.size(); j++){
      x_obs = TObs[j].x;
      y_obs = TObs[j].y;
      
      for(unsigned int k = 0; k < closestpred.size(); k++){
        if(closestpred[k].id == TObs[j].id){
          mu_x = closestpred[k].x;
          mu_y = closestpred[k].y;
      	  gauss_norm = 1.0 / (2.0 * M_PI * sig_x * sig_y);
 	  	  exponent = (pow(x_obs - mu_x, 2) / (2.0 * pow(sig_x, 2))) + (pow(y_obs - mu_y, 2) / (2.0 * pow(sig_y, 2)));
          particles[i].weight *= gauss_norm*exp(-exponent); 
        }
      }

    }
    weight_p += particles[i].weight;
  }
  
  for(unsigned int i = 0; i < particles.size(); i++){
    particles[i].weight /= weight_p;
    weights[i] = particles[i].weight;
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  default_random_engine gen;
  vector<Particle> rs_particle;
  std::uniform_int_distribution<int> particle_idx(0, num_particles -1);
  int index = particle_idx(gen);
  double beta = 0.0;
  double max_weight = 2.0* *max_element(weights.begin(), weights.end());
  
  for(unsigned int i =0 ;i < particles.size(); i++){
    std::uniform_real_distribution<double> random_dist(0.0, max_weight);
    double rn_dist = random_dist(gen);
    beta += rn_dist;
    
    while(beta > weights[index]){
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    rs_particle.push_back(particles[index]);
  }
  particles = rs_particle;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}