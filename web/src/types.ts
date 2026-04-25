export type Metrics = {
  task: string;
  total_samples: number;
  train_samples: number;
  val_samples: number;
  test_samples: number;
  defect_samples: number;
  good_samples: number;
  latency_ms: number;
  model_status: string;
};

export type ExampleItem = {
  id: string;
  category: string;
  status: string;
  image: string;
};

export type PredictResponse = {
  has_defect: boolean;
  confidence: number;
  latency_ms: number;
  overlay_url: string;
};
