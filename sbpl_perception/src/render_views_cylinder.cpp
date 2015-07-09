
#include <pcl/point_types.h>
//#include <pcl/apps/render_views_tesselated_sphere.h>
#include <sbpl_perception/render_views_cylinder.h>
#include <vtkCellData.h>
#include <vtkWorldPointPicker.h>
#include <vtkPropPicker.h>
#include <vtkPlatonicSolidSource.h>
#include <vtkLoopSubdivisionFilter.h>
#include <vtkTriangle.h>
#include <vtkTransform.h>
#if VTK_MAJOR_VERSION==6 || (VTK_MAJOR_VERSION==5 && VTK_MINOR_VERSION>4)
#include <vtkHardwareSelector.h>
#include <vtkSelectionNode.h>
#else 
#include <vtkVisibleCellSelector.h>
#endif
#include <vtkSelection.h>
#include <vtkCellArray.h>
#include <vtkTransformFilter.h>
#include <vtkCamera.h>
#include <vtkActor.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkPolyDataMapper.h>
#include <vtkPointPicker.h>
#include "vtkLight.h"
#include "vtkCylinderSource.h"
#include <vtkTransformPolyDataFilter.h>
#include <vtkRegularPolygonSource.h>
#include <vtkTexture.h>
#define PI 3.14159265


void
RenderViewsCylinder::generateViews() {
  //center object
  double CoM[3];
  vtkIdType npts_com = 0, *ptIds_com = NULL;
  vtkSmartPointer<vtkCellArray> cells_com = polydata_->GetPolys ();

  double center[3], p1_com[3], p2_com[3], p3_com[3], area_com, totalArea_com = 0;
  double comx = 0, comy = 0, comz = 0;
  for (cells_com->InitTraversal (); cells_com->GetNextCell (npts_com, ptIds_com);)
  {
    polydata_->GetPoint (ptIds_com[0], p1_com);
    polydata_->GetPoint (ptIds_com[1], p2_com);
    polydata_->GetPoint (ptIds_com[2], p3_com);
    vtkTriangle::TriangleCenter (p1_com, p2_com, p3_com, center);
    area_com = vtkTriangle::TriangleArea (p1_com, p2_com, p3_com);
    comx += center[0] * area_com;
    comy += center[1] * area_com;
    comz += center[2] * area_com;
    totalArea_com += area_com;
  }

  CoM[0] = comx / totalArea_com;
  CoM[1] = comy / totalArea_com;
  CoM[2] = comz / totalArea_com;

  vtkSmartPointer<vtkTransform> trans_center = vtkSmartPointer<vtkTransform>::New ();
  trans_center->Translate (-CoM[0], -CoM[1], -CoM[2]);
  vtkSmartPointer<vtkMatrix4x4> matrixCenter = trans_center->GetMatrix ();

  vtkSmartPointer<vtkTransformFilter> trans_filter_center = vtkSmartPointer<vtkTransformFilter>::New ();
  trans_filter_center->SetTransform (trans_center);
#if VTK_MAJOR_VERSION < 6
  trans_filter_center->SetInput (polydata_);
#else
  trans_filter_center->SetInputData (polydata_);
#endif
  trans_filter_center->Update ();

  //texture create
  vtkSmartPointer<vtkTexture> texture = vtkSmartPointer<vtkTexture>::New();
  
if (imgeformat_)
{
    texture->SetInputConnection(PNGReader_->GetOutputPort());
}
else
{
    texture->SetInputConnection(jPEGReader_->GetOutputPort());
}

  vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New ();
  mapper->SetInputConnection (trans_filter_center->GetOutputPort ());
  mapper->Update ();

  //scale so it fits in the mm or m kalyan
  double bb[6];
  mapper->GetBounds (bb);
  double ms = (std::max) ((std::fabs) (bb[0] - bb[1]),
                          (std::max) ((std::fabs) (bb[2] - bb[3]), (std::fabs) (bb[4] - bb[5])));
  double max_side = radius_circle_ / 2.0;
  double scale_factor = max_side / ms;
  //double scale_factor = 0.001;
  std::cout << "CAD Model Scaled by value: " << scale_factor << endl;
  
  vtkSmartPointer<vtkTransform> trans_scale = vtkSmartPointer<vtkTransform>::New ();
  trans_scale->Scale (scale_factor, scale_factor, scale_factor);
  vtkSmartPointer<vtkMatrix4x4> matrixScale = trans_scale->GetMatrix ();

  vtkSmartPointer<vtkTransformFilter> trans_filter_scale = vtkSmartPointer<vtkTransformFilter>::New ();
  trans_filter_scale->SetTransform (trans_scale);
  trans_filter_scale->SetInputConnection (trans_filter_center->GetOutputPort ());
  trans_filter_scale->Update ();

  mapper->SetInputConnection (trans_filter_scale->GetOutputPort ());
  mapper->Update ();

  //////////////////////////////
  // * Compute area of the mesh
  //////////////////////////////
  vtkSmartPointer<vtkCellArray> cells = mapper->GetInput ()->GetPolys ();
  vtkIdType npts = 0, *ptIds = NULL;

  double p1[3], p2[3], p3[3], area, totalArea = 0;
  for (cells->InitTraversal (); cells->GetNextCell (npts, ptIds);)
  {
    polydata_->GetPoint (ptIds[0], p1);
    polydata_->GetPoint (ptIds[1], p2);
    polydata_->GetPoint (ptIds[2], p3);
    area = vtkTriangle::TriangleArea (p1, p2, p3);
    totalArea += area;
  }

  

  //my version of camera positions to generate about a cirle
  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > cam_positions;
  cam_positions.resize (360);
  for (int i = 0; i < 360; i++)
  {
    
    float xcod = radius_circle_*cos(i*PI/ 180.0);
    float ycod = radius_circle_*sin(i*PI/ 180.0);
    float zcod = view_height_;
   cam_positions[i] = Eigen::Vector3f (xcod,ycod,zcod); 
  }


  
  double cam_pos[3];


  //create renderer and window
  vtkSmartPointer<vtkRenderWindow> render_win = vtkSmartPointer<vtkRenderWindow>::New ();
  vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New ();
  render_win->AddRenderer (renderer);
  render_win->SetSize (resolution_, resolution_);
  renderer->SetBackground (1.0, 1.0, 1.0);

  //adding lighting kalyan!
  vtkSmartPointer<vtkLight> light = vtkSmartPointer<vtkLight>::New();
  light->SetFocalPoint(1.875,0.6125,0);
  light->SetPosition(0.875,1.6125,1);
  //renderer->AddLight(light);


  //set global farplane clipping value for z in camera
  double farplanedist = 5;

  Eigen::Vector3f cam_pos_3f = cam_positions[0];
  Eigen::Vector3f perp = cam_pos_3f.cross (Eigen::Vector3f::UnitY ());
  
  
  //For each camera position, traposesnsform the object and render view
  for (size_t i = 0; i < cam_positions.size (); i++)
  {
    cam_pos[0] = cam_positions[i][0];
    cam_pos[1] = cam_positions[i][1];
    cam_pos[2] = cam_positions[i][2];


    //create temporal virtual camera
    vtkSmartPointer<vtkCamera> cam_tmp = vtkSmartPointer<vtkCamera>::New ();
    cam_tmp->SetViewAngle (view_angle_);
    cam_tmp->SetClippingRange(0.1, farplanedist);
    cam_tmp->SetViewUp (perp[0], perp[1], perp[2]);
    cam_tmp->SetPosition (cam_pos);
    cam_tmp->SetFocalPoint (0, 0, 0);
    cam_tmp->Modified ();


    //render view
    vtkSmartPointer<vtkActor> actor_view = vtkSmartPointer<vtkActor>::New ();
    actor_view->SetMapper (mapper);
    actor_view->SetTexture(texture);
    renderer->SetActiveCamera (cam_tmp);
    renderer->AddActor (actor_view);
    renderer->Modified ();
    //renderer->ResetCameraClippingRange ();
    render_win->Render ();

    //Depth window handles.
    vtkSmartPointer<vtkWindowToImageFilter> DepthwindowToImageFilter_;
    DepthwindowToImageFilter_ = vtkSmartPointer<vtkWindowToImageFilter>::New();
    DepthwindowToImageFilter_->SetInput(render_win);
    DepthwindowToImageFilter_->SetMagnification(1);
    DepthwindowToImageFilter_->SetInputBufferTypeToZBuffer();        //Extract z buffer value
    DepthwindowToImageFilter_->ReadFrontBufferOff(); // read from the back buffer
    DepthwindowToImageFilter_->Update();
    DepthArrwindowToImageFilter_.push_back (DepthwindowToImageFilter_);
    
    //RGB window handles!!!!
    vtkSmartPointer<vtkWindowToImageFilter> windowToImageFilter_;
    windowToImageFilter_ = vtkSmartPointer<vtkWindowToImageFilter>::New();
    windowToImageFilter_->SetInput(render_win);
    windowToImageFilter_->SetMagnification(1); //set the resolution of the output image (3 times the current resolution of vtk render window)
    windowToImageFilter_->SetInputBufferTypeToRGBA(); //also record the alpha (transparency) channel
    windowToImageFilter_->ReadFrontBufferOff(); // read from the back buffer
    windowToImageFilter_->Update();
    ArrwindowToImageFilter_.push_back (windowToImageFilter_);
    


    renderer->RemoveActor (actor_view);

    

    //create pose, from OBJECT coordinates to CAMERA coordinates!
    vtkSmartPointer<vtkTransform> transOCtoCC = vtkSmartPointer<vtkTransform>::New ();
    transOCtoCC->PostMultiply ();
    transOCtoCC->Identity ();
    transOCtoCC->Concatenate (cam_tmp->GetViewTransformMatrix ());

    //NOTE: vtk view coordinate system is different than the standard camera coordinates (z forward, y down, x right)
    //thus, the fliping in y and z
    vtkSmartPointer<vtkMatrix4x4> cameraSTD = vtkSmartPointer<vtkMatrix4x4>::New ();
    cameraSTD->Identity ();
    cameraSTD->SetElement (0, 0, 1);
    cameraSTD->SetElement (1, 1, -1);
    cameraSTD->SetElement (2, 2, -1);

    transOCtoCC->Concatenate (cameraSTD);
    transOCtoCC->Modified ();

    Eigen::Matrix4f pose_view;
    pose_view.setIdentity ();

    for (int x = 0; x < 4; x++)
      for (int y = 0; y < 4; y++)
        pose_view (x, y) = float (transOCtoCC->GetMatrix ()->GetElement (x, y));

    poses_.push_back (pose_view);

  }
}

