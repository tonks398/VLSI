include "globals.mzn";

% ==== Input variables ==== 

% Plate width and number of circuits
int: w;
int: n;

set of int: circ = 1..n;

% Width and height of circuits
array[circ] of int: cx;
array[circ] of int: cy;


% Plate bounds to find the height
int: lower;
int: upper;

% To specify whether a circuit is rotated or not
array[circ] of var bool: is_rot;

% Horizontal and Vertical coordinate of circuits
array[circ] of var 0..w: x;
array[circ] of var 0..upper: y;

% Height of the plate 
var lower..upper: h;

% A circuit cannot be rotated if its height is greater than plate width
constraint forall(i in circ)(cy[i] > w -> is_rot[i]==false);

% Actual horizontal and vertical dimension of circuits
array[circ] of var int: cxr;
cxr = [if is_rot[i] then cy[i] else cx[i] endif | i in circ];
array[circ] of var int: cyr;
cyr = [if is_rot[i] then cx[i] else cy[i] endif | i in circ];


%Domains
constraint forall(i in circ) (x[i]<=w-cxr[i])::domain;
constraint forall(i in circ) (y[i]<=h-cyr[i])::domain;


% Each circuit is an activity with duration equal to its height and amount of resource equal to its width
constraint cumulative(y, cyr, cxr, w);
constraint cumulative(x, cxr, cyr, h);

% Relationships among circuits
constraint diffn(x, y, cxr, cyr);

% ==== Symmetry breaking constraints ====

%THIS CAN BE COMMENTED OUT TO TRY WITH NO SYMMETRY BREAKING
% Circuits which have dimensions that are the same should not be rotated
constraint symmetry_breaking_constraint(
  forall (i in circ) (cx[i] = cy[i] -> is_rot[i] = 0)
);

% The biggest circuit is always placed in the bottom left part of the plate + under and on the left wrt the second biggest
array[circ] of int : ordered_circuits = sort_by(circ, [-cy[c]*cx[c] | c in circ]);

constraint symmetry_breaking_constraint(
  let {
    int: c1 = ordered_circuits[1], int: c2 = ordered_circuits[2]
  } in lex_lesseq([y[c1],x[c1]], [y[c2],x[c2]]) /\ x[c1] * 2 <= w /\ y[c1] * 2 <= h
); 
% THIS IS WHERE YOU STOP COMMENTING OUT THE SYMMETRY BREAKING CONSTRAINTS

% ==== Search ====

% Search and restart choice variables
int: search = 5;
% int: restart = 4;

% Search type
ann: search_ann = 
  if search == 1 then int_search([h], input_order, indomain_min)
  elseif search == 2 then int_search([h], first_fail, indomain_random)
  elseif search == 3 then int_search([h], input_order, indomain_min)
  elseif search == 4 then int_search([h], dom_w_deg, indomain_random)
  elseif search == 5 then int_search([h], first_fail, indomain_min)
  else int_search([h], dom_w_deg, indomain_random)
  endif;

%THIS IS THE RESTART FUNCTION
% Restart type
% ann: restart_ann =
%   if restart == 1 then restart_constant(300)
%   elseif restart == 2 then restart_linear(60)
%   elseif restart == 3 then restart_geometric(1.5, 200)
%   else restart_luby(150)
%   endif;

% Solve CP problem
%SWITCH THESE TWO WHEN RESTART IS USED
% solve :: search_ann :: restart_ann minimize h;
solve :: search_ann minimize h;



output ["\(w) \(h)\n\(n)\n"]
       %++ ["Plate height: \(height)\n"]
       ++ ["\(cxr[i]) \(cyr[i]) \(x[i]) \(y[i])"
       ++ "\n"
      | i in 1..n];