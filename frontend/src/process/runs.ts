export type Clip = {
  src: string
  step: string
  detail: string
}

export type Plot = {
  src: string
  title: string
  axes: string
}

export type Run = {
  id: string
  index: string
  nav: string
  title: string
  skim: string
  count: string
  countLabel: string
  clips: Clip[]
  clipNote?: string
  plots: Plot[]
  story: string[]
  next?: string
  limit?: { text: string; plot: Plot }
}

const root = "/process"

export const runs: Run[] = [
  {
    id: "cameras",
    index: "01",
    nav: "Cameras",
    title: "Cameras and joints",
    skim: "The first training run copies the expert’s arm motion from cameras and joint angles. The cube is never stacked.",
    count: "0 of 10",
    countLabel: "stacks on layouts the training set never saw",
    clips: [
      {
        src: `${root}/cameras-step2000.mp4`,
        step: "After 2,000 training steps",
        detail: "The policy grasps once and drops. The cube stays about where it started.",
      },
    ],
    plots: [
      {
        src: `${root}/cameras-loss.png`,
        title: "How closely the joints copy the demonstrations",
        axes: "Horizontal: training step. Vertical: loss. Lower means the joint targets match the demonstrations. A low loss is not a stack.",
      },
      {
        src: `${root}/cameras-success.png`,
        title: "What the arm actually did",
        axes: "Horizontal: training step. Vertical: count out of 10 held-out layouts. Stacks, grasps, and drops are drawn separately.",
      },
    ],
    story: [
      "This is the starting point. A scripted expert stacks the red cube on the green one, and it can see both cubes. Fifty of its successes become the training set. The learned policy is shown the top camera, the wrist camera, the sentence, and the six joint angles. It is not shown where the cubes are. Only the action expert is trained, the piece that outputs the next joint targets. The vision tower and the language model stay frozen, so their weights do not change.",
      "The flaw shows up when the two plots are read together. The loss is how closely those joint targets match the expert. It falls, and the validation loss ends near 0.15, so the arm motion does look like the demonstrations. On the robot, stacks stay at 0 for the whole run. The closest the red cube comes to sitting on the green one is 9.5 cm, which is the gap it was given at the start. Step 2,000 grasps once and drops. The arm has learned the gesture and not the task.",
    ],
    next: "The camera might already know where the cubes are, and the action expert might simply be ignoring it. That is the next measurement.",
  },
  {
    id: "readout",
    index: "02",
    nav: "Readout",
    title: "Where the cubes are",
    skim: "A small readout on the frozen camera finds each cube to about 1.6 cm. Training the camera on these demos makes that worse. The arm is not run.",
    count: "1.60 cm · 1.55 cm",
    countLabel: "median error for the red cube and the green cube",
    clips: [],
    plots: [
      {
        src: `${root}/see-localize.png`,
        title: "Error finding each cube from the frozen camera",
        axes: "Horizontal: training step of the vision tower. Vertical: median error on validation frames, in centimeters. The dashed line is the 1 cm bar. The dotted lines are the error from ignoring the image and guessing the average position.",
      },
    ],
    story: [
      "The question is whether the frozen vision features already contain the cube positions. A linear readout, one learned matrix, is fit on those features and asked for the x and y of each cube, using only frames where that cube is still on the floor. Guessing the average training position, with the image ignored, misses by about 8 cm. The frozen readout misses by 1.60 cm for red and 1.55 cm for green.",
      "That is much better than guessing, and it still misses a 1 cm bar set as the point where a grasp would be reliable. The vision tower was then trained on the same 50 demonstrations. No later step beat the untouched tower. The best checkpoint is step 0. At step 2,000 the red error is 5.05 cm and the green error is 2.33 cm, and the curve spends time worse than the no-image guess. The action expert was not trained here, so there is no robot clip.",
    ],
    next: "The positions are in the camera features, roughly, and the previous run’s action expert did not use them. The next run hands it the true positions.",
  },
  {
    id: "positions",
    index: "03",
    nav: "Positions",
    title: "Positions in the state",
    skim: "The true cube positions are written in next to the joint angles. The arm starts to grasp, and then to stack. Vision stays frozen.",
    count: "5 of 10",
    countLabel: "stacks at step 7,500, with 9 grasps and a best miss of 0.4 cm",
    clipNote: "Three checkpoints of this training. Each clip is the policy on one held-out layout.",
    clips: [
      {
        src: `${root}/positions-step1500.mp4`,
        step: "After 1,500 training steps",
        detail: "First grasps. Nothing stacks. The best reach is 1.4 cm, then a drop.",
      },
      {
        src: `${root}/positions-step5500.mp4`,
        step: "After 5,500 training steps",
        detail: "First stacks. 2 of 10. Grasps are now common.",
      },
      {
        src: `${root}/positions-step7500.mp4`,
        step: "After 7,500 training steps",
        detail: "5 of 10 stacks and 9 grasps. Best miss 0.4 cm.",
      },
    ],
    plots: [
      {
        src: `${root}/positions-success.png`,
        title: "Stacks appear only after grasps are common",
        axes: "Horizontal: training step, counted from the original start. Vertical: count out of 10. The run was paused at step 1,500 and continued with a new learning-rate schedule.",
      },
    ],
    story: [
      "Same demonstrations, same frozen vision and language. The difference is the state vector, the list of numbers passed in beside the image. After the six joint angles, the true x and y of the red cube and the green cube are written in, scaled by their training mean and spread. At test time those four numbers are the live positions. The policy no longer has to find the cubes in the image.",
      "The first 1,500 steps were cut short while the learning rate was already low. Grasps had just appeared: 0 stacks, and one layout reached 1.4 cm before a drop. Those weights were kept, the learning-rate schedule was restarted, and training ran 6,000 more steps. Stacks begin at step 5,500, 2 of 10. At step 7,500 the policy stacks 5 of 10, grasps on 9, and the best miss is 0.4 cm. It can recover a drop, because it replans from the live positions.",
    ],
    next: "This works for one sentence, and only because the positions are given. The next run asks the same weights to follow the other sentence.",
  },
  {
    id: "other",
    index: "04",
    nav: "Other sentence",
    title: "The other sentence",
    skim: "The positions stay. The sentence is the only thing that says which cube should move. The old skill disappears, and the new one never shows up.",
    count: "0 of 10",
    countLabel: "stacks for either sentence after 2,000 more steps",
    clipNote: "Both clips are “stack the red cube on the green cube,” before and after this training.",
    clips: [
      {
        src: `${root}/sentence-step0000.mp4`,
        step: "Before this training",
        detail: "The previous weights, on new layouts. Red on green stacks 2 of 10.",
      },
      {
        src: `${root}/sentence-step2000.mp4`,
        step: "After 2,000 training steps",
        detail: "Red on green is now 0 of 10. Green on red stayed at 0 throughout.",
      },
    ],
    plots: [
      {
        src: `${root}/sentence-success.png`,
        title: "Teaching the second sentence erases the first",
        axes: "Horizontal: training step. Vertical: stacks out of 10, drawn once for each sentence.",
      },
    ],
    story: [
      "Training starts from the weights that stacked 5 of 10 when told the positions. Both cube positions are still written into the state. The new data contains both sentences, “stack the red cube on the green cube” and the swap. Vision and language stay frozen, so the sentence can affect the action expert only through features that are not being trained.",
      "On the new test layouts, those starting weights still stack red on green 2 of 10, and green on red 0 of 10. The 2 is lower than the earlier 5 because the layouts are different and the policy samples noise, so a repeat is not exact. After 2,000 steps the red-on-green line has fallen to 0, and green on red has never left 0. The text of the sentence did not steer the arm. Continuing training from a frozen language model wiped out the skill that was already there.",
    ],
    next: "The action expert needs the sentence in the same place it needed the positions: as a few numbers in the state.",
  },
  {
    id: "both",
    index: "05",
    nav: "Both sentences",
    title: "Both sentences",
    skim: "A frozen classifier turns the sentence into two numbers and those numbers are written into the state. Both orders of the stack come back.",
    count: "7 of 10 · 6 of 10",
    countLabel: "red on green, then green on red, at step 8,000",
    clipNote: "The same checkpoint, one clip for each sentence.",
    clips: [
      {
        src: `${root}/both-red-step8000.mp4`,
        step: "After 8,000 training steps",
        detail: "“Stack the red cube on the green cube.” 7 of 10.",
      },
      {
        src: `${root}/both-green-step8000.mp4`,
        step: "After 8,000 training steps",
        detail: "“Stack the green cube on the red cube.” 6 of 10.",
      },
    ],
    plots: [
      {
        src: `${root}/both-success.png`,
        title: "Both sentences, after the learning rate is restarted",
        axes: "Horizontal: training step of this restart. Vertical: stacks out of 10. Red on green falls to 0 at step 2,000, then recovers.",
      },
    ],
    story: [
      "A small classifier is fit on the frozen language features and then frozen itself. It answers which cube is the one to pick and which is the one to leave, as two numbers. Those numbers are written into the state beside the cube positions. The action expert trains again. The learning rate is started over at the top of its schedule, because in the position run it had already decayed before the grasps became reliable.",
      "The path is not smooth. At the first check of this restart, red on green stacks 6 of 10 and green on red stacks 3 of 10. Red on green then drops to 0 at step 2,000, while green on red is at 2. By step 8,000 red on green is 7 of 10 and green on red is 6 of 10, with 9 grasps on each. Both sentences work once the sentence has been reduced to numbers the action expert can read.",
    ],
    limit: {
      text: "That classifier was only shown red and green. On held-out yellow and blue its accuracy is 0, for both the cube to move and the cube to leave. Training on those colors was not started. The bars below are the classifier, not the arm.",
      plot: {
        src: `${root}/both-accuracy.png`,
        title: "The sentence classifier does not know the held-out colors",
        axes: "Horizontal: train, validation, and held-out, for the source cube and the target cube. Vertical: accuracy, from 0 to 1. The dashed line is 0.9.",
      },
    },
    next: "Both sentences depend on being told the cube positions. The last run takes those positions away.",
  },
  {
    id: "again",
    index: "06",
    nav: "Cameras again",
    title: "Cameras again",
    skim: "The position numbers are removed. The policy has the cameras and the sentence only. A few layouts stack. Most are never grasped.",
    count: "3 of 10 · 1 of 10",
    countLabel: "red on green, then green on red, at step 8,000",
    clips: [
      {
        src: `${root}/cameras-again-step8000.mp4`,
        step: "After 8,000 training steps",
        detail: "“Stack the red cube on the green cube.” 3 of 10. Six layouts are never grasped.",
      },
    ],
    plots: [
      {
        src: `${root}/cameras-again-success.png`,
        title: "Stacks without being told the cube positions",
        axes: "Horizontal: training step. Vertical: stacks out of 10, one line for each sentence.",
      },
    ],
    story: [
      "The cube positions in the state are set back to zero. Vision and language stay frozen. After 8,000 steps, red on green stacks 3 of 10 and green on red stacks 1 of 10. The grasps that happen usually hold: red stacks 3 of its 4 grasps, and the best miss on red is 1.0 cm. The typical episode does not get that far. Six of the red layouts and seven of the green ones are never grasped, and the average miss stays at 7.3 cm and 9.1 cm.",
      "A follow-up trained the connector, the layer that compresses the camera into the tokens the language model reads, still with the positions removed. It ended at 2 of 10 and 2 of 10. Across the six runs, the arm stacks when the cube positions are written into the state. From the cameras alone, it usually does not find the cube.",
    ],
  },
]
