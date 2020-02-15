#! /usr/bin/env python
#
import math
import numpy as np
import random

def sphere_fibonacci_grid_points ( ng ):

  
  # phi = ( 1.0 + np.sqrt ( 5.0 ) ) / 2.0

  # theta = np.zeros ( ng )
  # sphi = np.zeros ( ng )
  # cphi = np.zeros ( ng )

  # for i in range ( 0, ng ):
  #   i2 = 2 * i - ( ng - 1 ) 
  #   theta[i] = 2.0 * np.pi * float ( i2 ) / phi
  #   sphi[i] = float ( i2 ) / float ( ng )
  #   cphi[i] = np.sqrt ( float ( ng + i2 ) * float ( ng - i2 ) ) / float ( ng )

  # xg = np.zeros ( ( ng, 3 ) )

  # for i in range ( 0, ng ) :
  #   xg[i,0] = cphi[i] * np.sin ( theta[i] )
  #   xg[i,1] = cphi[i] * np.cos ( theta[i] )
  #   xg[i,2] = sphi[i]

  # return xg
  

  rnd = 1.
  samples  = ng
  randomize = False
  if randomize:
      rnd = random.random() * samples

  points = []
  offset = 2./samples
  increment = math.pi * (3. - math.sqrt(5.));

  for i in range(samples):
      y = ((i * offset) - 1) + (offset / 2);
      r = math.sqrt(1 - pow(y,2))

      phi = ((i + rnd) % samples) * increment

      x = math.cos(phi) * r
      z = math.sin(phi) * r

      points.append([x,y,z])

  return np.array(points)


def sphere_fibonacci_grid_points_with_sym_metric (ng, half_whole):

  if (half_whole == 1):
    rnd = 1.
    samples  = ng
    randomize = False
    if randomize:
        rnd = random.random() * samples

    points = []
    offset = 2./samples
    increment = math.pi * (3. - math.sqrt(5.));

    for i in range(samples):
        y = ((i * offset) - 1) + (offset / 2);
        r = math.sqrt(1 - pow(y,2))

        phi = ((i + rnd) % samples) * increment

        x = math.cos(phi) * r
        z = math.sin(phi) * r

        points.append([x,y,z])

    return np.array(points)  
  else: # half_whole == 0
    rnd = 1.
    samples  = ng
    randomize = False
    if randomize:
        rnd = random.random() * samples

    points = []
    offset = 2./samples
    # offset = 1./samples
    increment = math.pi * (3. - math.sqrt(5.));

    # for i in range(math.ceil(samples/2)):
    for i in range(round(samples/2)):
        y = ((i * offset) - 1) + (offset / 2);
        r = math.sqrt(1 - pow(y,2))

        phi = ((i + rnd) % samples) * increment

        x = math.cos(phi) * r
        z = math.sin(phi) * r

        points.append([x,y,z])

    return np.array(points)
    
def sphere_fibonacci_grid_points_test ( ):

#*****************************************************************************80
#
#% SPHERE_FIBONACCI_GRID_POINTS_TEST tests SPHERE_FIBONACCI_GRID_POINTS.
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
  from r8mat_print_some import r8mat_print_some
  from r8mat_write import r8mat_write

  print ( '' )
  print ( 'SPHERE_FIBONACCI_GRID_POINTS_TEST' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  SPHERE_FIBONACCI_GRID_POINTS computes points on a sphere' )
  print ( '  that lie on a Fibonacci spiral.' )

  ng = 1000
  print ( '' )
  print ( '  Number of points NG = %d' % ( ng ) )

  xg = sphere_fibonacci_grid_points ( ng )

  r8mat_print_some ( ng, 3, xg, 0, 0, 19, 2, '  Part of the grid array:' )
#
#  Write the nodes to a file.
#
  filename = 'sphere_fibonacci_grid_points.xyz'

  r8mat_write ( filename, ng, 3, xg )
#
#  Terminate.
#
  print ( '' )
  print ( 'SPHERE_FIBONACCI_GRID_POINTS_TEST:' )
  print ( '  Normal end of execution.' )
  return

if ( __name__ == '__main__' ):
  from timestamp import timestamp
  timestamp ( )
  sphere_fibonacci_grid_points_test ( )
  timestamp ( )
