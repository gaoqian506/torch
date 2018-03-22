/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   showworld.cpp
 * Author: gq
 *
 * Created on March 22, 2018, 9:58 AM
 */

#include <cstdlib>
#include "flame.h"

using namespace std;

/*
 * 
 */
int main(int argc, char** argv) {
    
    torch::Flame flame;
//    flame.load("data/obj/cow.obj");
    flame.show();

    return 0;
}

