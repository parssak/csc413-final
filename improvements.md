# Improvements

## Not trading enough
Tends to learn to buy and hold.

=== Potential Solutions === 

Option A:
  - Negative reward for having a prolonged period of non-actions
      - If reaching a distance of N from the current to previous action,
        and is remaining to not perform a given action, apply

      ```
      e.g.

        S - - - - ... N steps ... - (curr)
        -> If not performing an action at curr, and previous [TODO @parssa]
      ```

Option B:
  - Raising exploration rate

## "Burst Trading"
Makes several trades in short period of time, 
with long pauses in between those bursts.

=== Potential Solutions ===

Option A:
  - Negative reward for incredibly high frequency in a rolling window
     - Multiply by ratio in the reward if performing an action when 
       majority of trades in the last N timesteps have also been actions

      ```
      e.g.
        S L L S - - S - (curr)

        -> If perform action at curr, where the previous n=8 timesteps had 5 actions, then apply a multiplier that diminishes reward by 5/8 = 0.625
      ```

Option B:
- Raising exploration rate


## Improve Generation
Current generation technique is very naive.
- Tends to become too smooth
- Not always representative of input data
- Hit or miss

=== Potential Solutions ===

Option A:
  Use GANs
     - Look into GANs further...

## Not following trends
Model does not take into account no trend, up trend and down trend
[TODO]: Add solutions

=== Potential Solutions ===
we can have a model that predicts where the future HH and LL
will be and pass that into the model during testing

LSTM generates predictions on future HH and LL

Model takes in distance to closest significant point

  e.g.

  HH - - - -3 (curr) - - - - - 
  - - +2 - - LL (curr - - - - 








