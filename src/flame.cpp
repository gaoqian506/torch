/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   flame.cpp
 * Author: gq
 * 
 * Created on March 22, 2018, 9:46 AM
 */

#include "flame.h"
#include <GL/glew.h>
#include <GL/glut.h>
#include "path_tracer.h"
#include <optixu/optixu_math_namespace.h> // make float3





namespace torch {
    
    Flame* g_flame = 0;
    
    
Flame::Flame() {
    
    width_ = 512;
    height_ = 512;

    init_glut();
    create_context();
    init_world();
    setup_camera();
    context_->validate();
    mesh_.context = context_;
    g_flame = this;

    
}

void Flame::init_glut() {
    
    int argc = 1; 
    glutInit( &argc, 0 );
    glutInitDisplayMode( GLUT_RGB | GLUT_ALPHA | GLUT_DEPTH | GLUT_DOUBLE );
    glutInitWindowSize( width_, height_ );
    glutInitWindowPosition( 100, 100 );                                               
    glutCreateWindow( "torch" );
    glutHideWindow();  
    glewInit();
}


void Flame::create_context() {
    
    int rr_begin_depth = 1;
    bool use_pbo = true;
    int sqrt_num_samples = 2;    

    context_ = optix::Context::create();
    context_->setRayTypeCount( 2 );
    context_->setEntryPointCount( 1 );
    context_->setStackSize( 1800 );
    
    context_[ "scene_epsilon"                  ]->setFloat( 1.e-3f );
    context_[ "pathtrace_ray_type"             ]->setUint( 0u );
    context_[ "pathtrace_shadow_ray_type"      ]->setUint( 1u );
    context_[ "rr_begin_depth"                 ]->setUint( rr_begin_depth );   
    
    optix::Buffer buffer = sutil::createOutputBuffer( context_, RT_FORMAT_FLOAT4, width_, height_, use_pbo );
    context_["output_buffer"]->set( buffer );
    
    // Setup programs
    
    optix::Program ray_gen_program = context_->createProgramFromPTXFile("ptx/path_tracer.ptx", "pathtrace_camera" );
    context_->setRayGenerationProgram( 0, ray_gen_program );
    optix::Program except_program = context_->createProgramFromPTXFile("ptx/path_tracer.ptx", "exception" );
    context_->setExceptionProgram( 0, except_program );
    optix::Program miss_program = context_->createProgramFromPTXFile("ptx/path_tracer.ptx", "miss" );
    context_->setMissProgram( 0, except_program );        
    
    context_[ "sqrt_num_samples" ]->setUint( sqrt_num_samples );
    context_[ "bad_color"        ]->setFloat( 1000000.0f, 0.0f, 1000000.0f ); // Super magenta to make sure it doesn't get averaged out in the progressive rendering.
    context_[ "bg_color"         ]->setFloat( optix::make_float3(0.0f) );  
    
}

void Flame::init_world() {
    
    ParallelogramLight light;
    light.corner   = optix::make_float3( 343.0f, 548.6f, 227.0f);
    light.v1       = optix::make_float3( -130.0f, 0.0f, 0.0f);
    light.v2       = optix::make_float3( 0.0f, 0.0f, 105.0f);
    light.normal   = optix::normalize( cross(light.v1, light.v2) );
    light.emission = optix::make_float3( 15.0f, 15.0f, 5.0f );

    optix::Buffer light_buffer = context_->createBuffer( RT_BUFFER_INPUT );
    light_buffer->setFormat( RT_FORMAT_USER );
    light_buffer->setElementSize( sizeof( ParallelogramLight ) );
    light_buffer->setSize( 1u );
    memcpy( light_buffer->map(), &light, sizeof( light ) );
    light_buffer->unmap();
    context_["lights"]->setBuffer( light_buffer );    
    
   // Set up material
    optix::Material diffuse = context_->createMaterial();
    optix::Program diffuse_ch = context_->createProgramFromPTXFile("ptx/path_tracer.ptx", "diffuse" );
    optix::Program diffuse_ah = context_->createProgramFromPTXFile("ptx/path_tracer.ptx", "shadow" );
    diffuse->setClosestHitProgram( 0, diffuse_ch );
    diffuse->setAnyHitProgram( 1, diffuse_ah );    
    
    optix::Material diffuse_light = context_->createMaterial();
    optix::Program diffuse_em = context_->createProgramFromPTXFile("ptx/path_tracer.ptx", "diffuseEmitter" );
    diffuse_light->setClosestHitProgram( 0, diffuse_em );    
    
    // Set up parallelogram programs
    optix::Program pgram_bounding_box = context_->createProgramFromPTXFile( "ptx/parallelogram.ptx", "bounds" );
    optix::Program pgram_intersection = context_->createProgramFromPTXFile( "ptx/parallelogram.ptx", "intersect" );   
    
    // Floor
    optix::Geometry floor_geometry = context_->createGeometry();
    floor_geometry->setPrimitiveCount( 1u );
    floor_geometry->setIntersectionProgram( pgram_intersection );
    floor_geometry->setBoundingBoxProgram( pgram_bounding_box );
    {
        optix::float3 anchor = optix::make_float3( 0.0f, 0.0f, 0.0f );
        optix::float3 offset1 = optix::make_float3( 0.0f, 0.0f, 559.2f );
        optix::float3 offset2 = optix::make_float3( 556.0f, 0.0f, 0.0f );

        optix::float3 normal = normalize( cross( offset1, offset2 ) );
        float d = dot( normal, anchor );
        optix::float4 plane = make_float4( normal, d );

        optix::float3 v1 = offset1 / dot( offset1, offset1 );
        optix::float3 v2 = offset2 / dot( offset2, offset2 );

        floor_geometry["plane"]->setFloat( plane );
        floor_geometry["anchor"]->setFloat( anchor );
        floor_geometry["v1"]->setFloat( v1 );
        floor_geometry["v2"]->setFloat( v2 );
    }
    
    const optix::float3 white = optix::make_float3( 0.8f, 0.8f, 0.8f );
    optix::GeometryInstance floor_gi = context_->createGeometryInstance();
    floor_gi->setGeometry(floor_geometry);
    floor_gi->addMaterial(diffuse);
    floor_gi["diffuse_color"]->setFloat(white);
    
    // Light
    optix::Geometry light_geometry = context_->createGeometry();
    light_geometry->setPrimitiveCount( 1u );
    light_geometry->setIntersectionProgram( pgram_intersection );
    light_geometry->setBoundingBoxProgram( pgram_bounding_box );
    {
        optix::float3 anchor = optix::make_float3( 0.0f, 0.0f, 0.0f );
        optix::float3 offset1 = optix::make_float3( 0.0f, 0.0f, 559.2f );
        optix::float3 offset2 = optix::make_float3( 556.0f, 0.0f, 0.0f );

        optix::float3 normal = normalize( cross( offset1, offset2 ) );
        float d = dot( normal, anchor );
        optix::float4 plane = make_float4( normal, d );

        optix::float3 v1 = offset1 / dot( offset1, offset1 );
        optix::float3 v2 = offset2 / dot( offset2, offset2 );

        light_geometry["plane"]->setFloat( plane );
        light_geometry["anchor"]->setFloat( anchor );
        light_geometry["v1"]->setFloat( v1 );
        light_geometry["v2"]->setFloat( v2 );
    }    
    const optix::float3 light_em = optix::make_float3( 15.0f, 15.0f, 5.0f );
    optix::GeometryInstance light_gi = context_->createGeometryInstance();
    light_gi->setGeometry(floor_geometry);
    light_gi->addMaterial(diffuse_light);
    light_gi["emission_color"]->setFloat(light_em);
    
    
    // Create geometry group
    optix::GeometryGroup geometry_group = context_->createGeometryGroup();
    geometry_group->setAcceleration( context_->createAcceleration( "Trbvh" ) );
    geometry_group->addChild(floor_gi);
    geometry_group->addChild(light_gi);
    context_["top_object"]->set( geometry_group );    
    
}

void Flame::setup_camera() {

    optix::float3 camera_eye    = optix::make_float3( 278.0f, 273.0f, -900.0f );
    optix::float3 camera_lookat = optix::make_float3( 278.0f, 273.0f,    0.0f );
    optix::float3 camera_up     = optix::make_float3(   0.0f,   1.0f,    0.0f );

    optix::Matrix4x4 camera_rotate  = optix::Matrix4x4::identity();
    
    const float fov  = 35.0f;
    const float aspect_ratio = static_cast<float>(width_) / static_cast<float>(height_);
    
    optix::float3 camera_u, camera_v, camera_w;
    sutil::calculateCameraVariables(
            camera_eye, camera_lookat, camera_up, fov, aspect_ratio,
            camera_u, camera_v, camera_w, /*fov_is_vertical*/ true );

    const optix::Matrix4x4 frame = optix::Matrix4x4::fromBasis( 
            optix::normalize( camera_u ),
            optix::normalize( camera_v ),
            optix::normalize( -camera_w ),
            camera_lookat);
    const optix::Matrix4x4 frame_inv = frame.inverse();
    // Apply camera rotation twice to match old SDK behavior
    const optix::Matrix4x4 trans     = frame*camera_rotate*camera_rotate*frame_inv; 

    camera_eye    = make_float3( trans*make_float4( camera_eye,    1.0f ) );
    camera_lookat = make_float3( trans*make_float4( camera_lookat, 1.0f ) );
    camera_up     = make_float3( trans*make_float4( camera_up,     0.0f ) );

    sutil::calculateCameraVariables(
            camera_eye, camera_lookat, camera_up, fov, aspect_ratio,
            camera_u, camera_v, camera_w, true );

    camera_rotate = optix::Matrix4x4::identity();

    int frame_number = 1;
//    bool camera_changed = false;
//    if( camera_changed ) // reset accumulation
//        frame_number = 1;
//    camera_changed = false;    
    
    context_[ "frame_number" ]->setUint( frame_number++ );
    context_[ "eye"]->setFloat( camera_eye );
    context_[ "U"  ]->setFloat( camera_u );
    context_[ "V"  ]->setFloat( camera_v );
    context_[ "W"  ]->setFloat( camera_w );    
}

void Flame::load(const std::string& scene_file) {
    
    loadMesh(scene_file, mesh_);
    
    //optix::Aabb aabb;
    //aabb.set( mesh.bbox_min, mesh.bbox_max );

//    optix::GeometryGroup geometry_group = context_->createGeometryGroup();
//    geometry_group->addChild( mesh.geom_instance );
//    geometry_group->setAcceleration( context_->createAcceleration( "Trbvh" ) );
//    context_[ "top_object"   ]->set( geometry_group ); 
//    context_[ "top_shadower" ]->set( geometry_group );  
    
}

void Flame::show() {

    
    // Initialize GL state                                                            
    glMatrixMode(GL_PROJECTION);                                                   
    glLoadIdentity();                                                              
    glOrtho(-1, 1, -1, 1, -1, 1 );                                                   

    glMatrixMode(GL_MODELVIEW);                                                    
    glLoadIdentity();                                                              

    glViewport(0, 0, width_, height_);

    glutShowWindow();                                                              
    //glutReshapeWindow( width_, height_);

    // register glut callbacks
    glutDisplayFunc( Flame::display );
//    glutIdleFunc( glutDisplay );
//    glutReshapeFunc( glutResize );
//    glutKeyboardFunc( glutKeyboardPress );
//    glutMouseFunc( glutMousePress );
//    glutMotionFunc( glutMouseMotion );

//    registerExitHandler();

    glutMainLoop();
    
}

void Flame::launch() {
    context_->launch( 0, width_, height_ );
}

optix::Buffer Flame::output_buffer() {
    return context_[ "output_buffer" ]->getBuffer();
}

void Flame::display() {
    
    //updateCamera();
    if (g_flame) {
        g_flame->launch();
        sutil::displayBufferGL( g_flame->output_buffer() );
    }
    



//    {
//      static unsigned frame_count = 0;
//      sutil::displayFps( frame_count++ );
//    }

    glutSwapBuffers();
    
}


} // namespace torch

/*
 * 
 * //#include <optixu_math_namespace.h>
//#include <optixu/optixu_math_stream_namespace.h>

//#include <optixu/optixpp_namespace.h>
//#include <optixu/optixu_math_stream_namespace.h>
//using namespace optix;
 * 
 * 
 *     
 * 
 *     
    optix::GeometryGroup geometry_group = context_->createGeometryGroup();
    //geometry_group->addChild( mesh_.geom_instance );
    geometry_group->setAcceleration( context_->createAcceleration( "Trbvh" ) );
    context_[ "top_object"   ]->set( geometry_group ); 
    context_[ "top_shadower" ]->set( geometry_group );     
    int argc = 1;
    glutInit( &argc, 0 );
    glutInitDisplayMode( GLUT_RGB | GLUT_ALPHA | GLUT_DEPTH | GLUT_DOUBLE );
    glutInitWindowSize( width_, height_ );
    glutInitWindowPosition( 100, 100 );                                               
    glutCreateWindow( "torch" );
    //glutHideWindow();
 * 
 * 
//-------------------
//void glutInitialize( int* argc, char** argv )
//{
//    glutInit( argc, argv );
//    glutInitDisplayMode( GLUT_RGB | GLUT_ALPHA | GLUT_DEPTH | GLUT_DOUBLE );
//    glutInitWindowSize( width, height );
//    glutInitWindowPosition( 100, 100 );                                               
//    glutCreateWindow( SAMPLE_NAME );
//    glutHideWindow();                                                              
//}

 * 
 *     GLenum r = glewInit();
    if (r = GLEW_OK) {
        
    }
    GLuint vbo = 0;

flame::flame(const flame& orig) {
}

flame::~flame() {
}
 
 */

