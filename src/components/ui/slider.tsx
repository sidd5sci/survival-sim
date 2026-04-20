import type { ChangeEvent } from "react";

type SliderProps = {
  value: number[];
  min: number;
  max: number;
  step?: number;
  onValueChange: (value: number[]) => void;
};

export function Slider({ value, min, max, step = 1, onValueChange }: SliderProps) {
  const currentValue = value[0] ?? min;

  const handleChange = (event: ChangeEvent<HTMLInputElement>) => {
    onValueChange([Number(event.target.value)]);
  };

  return <input type="range" className="w-full accent-blue-500" value={currentValue} min={min} max={max} step={step} onChange={handleChange} />;
}
