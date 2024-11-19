---
layout: post
author: Courtney McBeth
categories: ['Basics of Sailing']
logo: algo_logo.png
title: Terminology
---

<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

## Velocity Made Good (VMG)
<p>
In sailing, velocity made good (VMG) refers to the speed at which the sailboat is moving in a given direction. Commonly, this direction is either directly upwind or downwind. In our implementation, this direction is toward the target waypoint. We define velocity made good as the dot product of the boat's velocity vector, \(\vec{v_b}\) and the unit vector representing the difference between the target position and the boat position, \(\vec{t_0}\).
\[vmg = \vec{v_b} \cdot \vec{t_0} \]
</p>

In the example below, the difference between absolute velocity and velocity made good can be seen. Here, the target is directly upwind.

![VMG]({{ site.baseurl }}/images/vmg.jpg "VMG")

## Polar Diagram

A polar diagram (or polar plot) is used to determine the maximum velocity of a sailboat for a given apparent wind direction and wind speed. Here, the true wind angle is the wind angle experienced by the boat (or the relative wind direction).

![Polar Diagram]({{ site.baseurl }}/images/polar_diagram.png "Polar Diagram")

Each sailboat has its own unique polar diagram based on its size and shape. For most sailboats, a simplified polar diagram that does not rely on wind speed can be used without experiencing much difference.

## Points of Sailing

The points of sailing refer to the direction that a boat is sailing relative to the wind. Points where the wind is on the starboard (right) side of the boat are _starboard tacks_ (green below) and points where the wind is on the port (left) side of the boat are _port tacks_ (red below). Sailing directly upwind is not possible, so the region between starboard and port close hauled is often referred to as the "no go zone".

![Points of Sailing]({{ site.baseurl }}/images/points_of_sailing.jpg "Points of Sailing")

## Angle of Attack

Angle of attack refers to the angle between the wind and the sail. Each point of sail requires its own angle of attack. When the boat is close hauled, the sail will be _in tight_ on the boat. As the point of sail progresses toward a dead run, the sail should be put further and further out.

![Angle of Attack]({{ site.baseurl }}/images/angle_of_attack.jpg "Angle of Attack")

As the angle of attack increases, the sail progresses from being _attached_ (lift is maximized) to _separated_ and then _stalled_ (drag is maximized).

![Lift and Drag]({{ site.baseurl }}/images/lift_drag.png "Lift and Drag")

## Tacking and Jibing

Tacking and jibing are both sailing maneuvers which involve switching from a starboard to a port tack (or vice versa). In tacking, the boat is sailing upwind, while in jibing, the boat is sailing downwind.

![Tacking and Jibing]({{ site.baseurl }}/images/tacking_jibing.gif "Tacking and Jibing")

## Beating
<p>
Since no sailboat can sail directly upwind, the boat must sail in a zig-zag path while consistently tacking to make progress in this direction. This is called beating. The <i>beating parameter</i> (often \(p_c\)) refers to the lateral (side-to-side) distance that the boat can sail before tacking (1/2 of the total width of the path).
</p>

![Beating]({{ site.baseurl }}/images/beating.png "Beating")