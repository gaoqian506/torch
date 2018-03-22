/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   flame.h
 * Author: gq
 *
 * Created on March 22, 2018, 9:46 AM
 */

#ifndef TORCH_FLAME_H
#define TORCH_FLAME_H

#include <string>
#include <OptiXMesh.h>

namespace torch {
 
class Flame {

public:
    Flame();
    void load(const std::string& scene_file);
    void show();
private:
    
    void init_glut();
    void create_context();
    void init_world();
    void setup_camera();
    void launch();
    optix::Buffer output_buffer();    
    static void display();
    
    
    int width_;
    int height_;
    optix::Context context_;
    OptiXMesh mesh_;
};

   
} // namespace torch


#endif /* TORCH_FLAME_H */

