"use client";

import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

type Point = {
  trading_date_ist: string;
  touch_watch_calibration_error: number;
  avoidance_calibration_error: number;
  options_calibration_error: number;
};

export function CalibrationTimeSeries({ data }: { data: Point[] }) {
  return (
    <div className="w-full h-80">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart
          data={data}
          margin={{ top: 12, right: 20, left: 12, bottom: 12 }}
        >
          <CartesianGrid
            stroke="#23262B"
            strokeDasharray="2 4"
            vertical={false}
          />
          <XAxis
            dataKey="trading_date_ist"
            tick={{ fill: "#5E6168", fontSize: 11 }}
            stroke="#23262B"
            tickFormatter={(v: string) => v.slice(5)}
            minTickGap={36}
          />
          <YAxis
            tick={{ fill: "#5E6168", fontSize: 11 }}
            stroke="#23262B"
            tickFormatter={(v: number) =>
              (v >= 0 ? "+" : "") + (v * 100).toFixed(0) + "%"
            }
            domain={[-0.18, 0.18]}
          />
          <ReferenceLine y={0.08} stroke="#C66B5C" strokeDasharray="3 3" />
          <ReferenceLine y={-0.08} stroke="#C66B5C" strokeDasharray="3 3" />
          <ReferenceLine y={0} stroke="#5E6168" />
          <Tooltip
            contentStyle={{
              background: "#15171A",
              border: "1px solid #23262B",
              borderRadius: 2,
              fontSize: 12,
            }}
            labelStyle={{ color: "#9DA0A6" }}
            itemStyle={{ color: "#E8E6E1" }}
            formatter={(v: number) =>
              (v >= 0 ? "+" : "") + (v * 100).toFixed(2) + "%"
            }
          />
          <Legend
            wrapperStyle={{ fontSize: 12, color: "#9DA0A6" }}
            iconType="plainline"
          />
          <Line
            name="Touch watch"
            type="monotone"
            dataKey="touch_watch_calibration_error"
            stroke="#7C9BB8"
            strokeWidth={1.5}
            dot={false}
            isAnimationActive={false}
          />
          <Line
            name="Avoidance"
            type="monotone"
            dataKey="avoidance_calibration_error"
            stroke="#9FB6CB"
            strokeWidth={1.5}
            dot={false}
            isAnimationActive={false}
          />
          <Line
            name="Options"
            type="monotone"
            dataKey="options_calibration_error"
            stroke="#7DA982"
            strokeWidth={1.5}
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
