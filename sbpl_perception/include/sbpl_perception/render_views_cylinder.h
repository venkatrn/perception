/*
 * render_views_tesselated_sphere.h
 *
 *  Created on: Dec 23, 2011
 *      Author: aitor
 */

#ifndef RENDER_VIEWS_CYLINDER_H_
#define RENDER_VIEWS_CYLINDER_H_


#include <pcl/common/common.h>
#include <boost/function.hpp>


#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkSmartPointer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkPolyData.h>
#include <vtkSphereSource.h>
#include <vtkWindowToImageFilter.h>
#include <vtkPNGWriter.h>
#include <vtkJPEGReader.h>
#include <vtkPNGReader.h>
#include <vtkTexture.h>
#include <vtkImageShiftScale.h>
#include <vtkBMPWriter.h>
#include <vtkJPEGWriter.h>
#include <math.h> 
#include <vtkImageWriter.h>
#include <vtkTIFFWriter.h>

    /** \brief @b Class to render synthetic views of a 3D mesh using a tesselated sphere
     * NOTE: This class should replace renderViewTesselatedSphere from pcl::visualization.
     * Some extensions are planned in the near future to this class like removal of duplicated views for
     * symmetrical objects, generation of RGB synthetic clouds when RGB available on mesh, etc.
     * \author Aitor Aldoma
     * \ingroup apps
     */
    class RenderViewsCylinder
    {
    private:
      std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > poses_;
      std::vector<vtkSmartPointer<vtkWindowToImageFilter> > ArrwindowToImageFilter_;
      std::vector<vtkSmartPointer<vtkWindowToImageFilter> > DepthArrwindowToImageFilter_;


      int NumViews_;
      int resolution_;
      float view_angle_;
      float radius_circle_;
      bool imgeformat_;
      float view_height_;
      double focus_[3];
      float offset_;


      vtkSmartPointer<vtkPolyData> polydata_;
      vtkSmartPointer<vtkJPEGReader> jPEGReader_;
      vtkSmartPointer<vtkPNGReader> PNGReader_;


    public:
      RenderViewsCylinder ()
      {
        resolution_ = 150;
        view_angle_ = 57;
        radius_circle_ = 1.f;
        imgeformat_ = true;
        view_height_ = 0.f;
        //focus_.iners = {0,0,0};
        focus_ [0] = 0;  
        focus_ [1] = 0;  
        focus_ [2] = 0; 
        offset_ = 0; 
        NumViews_ = 360;
      }



      /* \brief Sets the size of the render window
       * \param res resolution size
       */
      void
      setResolution (int res)
      {
        resolution_ = res;
      }

          /* \
       * \Set Yaw offset
       */
      void
      setNumViewsCircle (int views)
      {
        NumViews_ = views;
      }

      /* \
       * \Set Yaw offset
       */
      void
      setYawOffset ( float offset)
      {
        offset_ = offset;
      }


      /* \
       * \Set focus of camera
       */
      void
      setFocusPoint ( double focus[3])
      {
        focus_[0] = focus[0];
        focus_[1] = focus[1];
        focus_[2] = focus[2];
      }


      /* \brief Radius of the circle where the virtual camera will be placed
       * \param use true indicates to use vertices, false triangle centers
       */
      void
      setRadiusCircle (float radius)
      {
        radius_circle_ = radius;
      }

      /* \brief height of the circle where the virtual camera will be placed
       * \param use true indicates to use vertices, false triangle centers
       */
      void
      setHeightCircle (float height)
      {
        view_height_ = height;
      }



      /* \brief Sets the view angle of the virtual camera
       * \param angle view angle in degrees
       */
      void
      setViewAngle (float angle)
      {
        view_angle_ = angle;
      }

      /* \brief adds the mesh to be used as a vtkPolyData
       * \param polydata vtkPolyData object
       */
      void
      addModelFromPolyData (vtkSmartPointer<vtkPolyData> &polydata)
      {
        polydata_ = polydata;
      }

      void
      setPNGImageFormat (bool use)
      {
        imgeformat_ = use;
      }

      // add jpeg file
      void
      addModelJpegImage (vtkSmartPointer<vtkJPEGReader> &ImageFile)
      {
        jPEGReader_ = ImageFile;
      }
      // add png file
      void
      addModelPNGImage (vtkSmartPointer<vtkPNGReader> &ImageFile)
      {
        PNGReader_ = ImageFile;
      }
 
      /* \brief performs the rendering and stores the generated information
       */
      void
      generateViews ();

      /* \brief Get the generated poses for the generated views
       * \param poses 4x4 matrices representing the pose of the cloud relative to the model coordinate system
       */
      void
      getPoses (std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > & poses)
      {
        poses = poses_;
      }

      /* 
       * \Get pointer to windows that need to be saved
       */
      void
      getWindows (std::vector<vtkSmartPointer<vtkWindowToImageFilter> > & windows)
      {
        windows = ArrwindowToImageFilter_;
      }
      //depth window pointers
      void
      getDepthWindows (std::vector<vtkSmartPointer<vtkWindowToImageFilter> > & windows)
      {
        windows = DepthArrwindowToImageFilter_;
      }
    };


#endif 
