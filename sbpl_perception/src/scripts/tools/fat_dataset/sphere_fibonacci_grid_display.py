#! /usr/bin/env python
#
def sphere_fibonacci_grid_display ( ng, xg, filename ):

#
  import matplotlib.pyplot as plt
  import numpy as np
  from mpl_toolkits.mplot3d import Axes3D

  fig = plt.figure ( )
  ax = fig.add_subplot ( 111, projection = '3d' )
#
#  Draw the grid points.
#
  ax.scatter ( xg[:,0], xg[:,1], xg[:,2], 'b' )

  ax.set_xlabel ( '<---X--->' )
  ax.set_ylabel ( '<---Y--->' )
  ax.set_zlabel ( '<---Z--->' )
  ax.set_title ( 'Fibonacci spiral on sphere' )
  ax.grid ( True )
  ax.axis ( 'equal' )

# plt.show ( )
  plt.savefig ( filename )

  plt.clf ( )

  return

def sphere_fibonacci_grid_display_test ( ):

#*****************************************************************************80
#
#% SPHERE_FIBONACCI_GRID_DISPLAY_TEST tests SPHERE_FIBONACCI_GRID_DISPLAY.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    16 May 2015
#
#  Author:
#
#    John Burkardt
#
  import platform
  from sphere_fibonacci_grid_points import sphere_fibonacci_grid_points

  print ( '' )
  print ( 'SPHERE_FIBONACCI_GRID_DISPLAY_TEST' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  SPHERE_FIBONACCI_GRID_DISPLAY displays points on a sphere' )
  print ( '  that lie on a Fibonacci spiral.' )

  ng = 1000
  print ( '' )
  print ( '  Number of points NG = %d' % ( ng ) )

  xg = sphere_fibonacci_grid_points ( ng )
#
#  Display the nodes.
#
  filename = 'sphere_fibonacci_grid_display.png'

  sphere_fibonacci_grid_display ( ng, xg, filename )

  print ( '' )
  print ( '  Plot saved to file "%s".' % ( filename ) )
#
#  Terminate.
#
  print ( '' )
  print ( 'SPHERE_FIBONACCI_GRID_DISPLAY_TEST:' )
  print ( '  Normal end of execution.' )
  return

if ( __name__ == '__main__' ):
  # from timestamp import timestamp
  # timestamp ( )
  sphere_fibonacci_grid_display_test ( )
  # timestamp ( )
