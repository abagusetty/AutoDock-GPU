/*

AutoDock-GPU, an OpenCL implementation of AutoDock 4.2 running a Lamarckian Genetic Algorithm
Copyright (C) 2017 TU Darmstadt, Embedded Systems and Applications Group, Germany. All rights reserved.
For some of the code, Copyright (C) 2019 Computational Structural Biology Center, the Scripps Research Institute.

AutoDock is a Trade Mark of the Scripps Research Institute.

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

*/


#ifndef PLATFORMS_H
#define PLATFORMS_H

#include "commonMacros.h"

/* Get all available platforms
   Inputs:
        none
   Outputs:
        platform_id -
        platformCount -
*/
int getPlatforms(
                 cl_platform_id** platforms_id,
                 cl_uint*         platformCount
                );

/* Get all platforms' attributes
   Inputs:
        none
   Outputs:
        platform_id -
        platformCount -
*/
int getPlatformAttributes(
                          cl_platform_id* platform_id,
                          cl_uint         platformCount
                         );

#endif /* PLATFORMS_H */
