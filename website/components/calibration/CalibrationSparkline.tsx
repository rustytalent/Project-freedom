"use client";

import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export type SparklinePoint = {
  trading_date_ist: string;
  touch_watch_calibration_error: number;
};

export function CalibrationSparkline({
  data,
  height = 220,
  showAxes = true,
  animate = false,
  animationBeginMs = 0,
  animationDurationMs = 900,
}: {
  data: SparklinePoint[];
  height?: number;
  showAxes?: boolean;
  /** When true, the line draws in on mount via Recharts animation. */
  animate?: boolean;
  /** Delay before the draw begins (used for hero choreography). */
  animationBeginMs?: number;
  /** Total draw duration. */
  animationDurationMs?: number;
}) {
  return (
    <div className="w-full" style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart
          data={data}
          margin={{ top: 8, right: 12, left: 8, bottom: 8 }}
        >
          {showAxes && (
            <CartesianGrid
              stroke="#23262B"
              strokeDasharray="2 4"
              vertical={false}
            />
          )}
          {showAxes && (
            <XAxis
              dataKey="trading_date_ist"
              tick={{ fill: "#5E6168", fontSize: 11 }}
              stroke="#23262B"
              tickFormatter={(v: string) => v.slice(5)}
              minTickGap={32}
            />
          )}
          {showAxes && (
            <YAxis
              tick={{ fill: "#5E6168", fontSize: 11 }}
              stroke="#23262B"
              tickFormatter={(v: number) =>
                (v >= 0 ? "+" : "") + (v * 100).toFixed(0) + "%"
              }
              domain={[-0.12, 0.12]}
            />
          )}
          <ReferenceLine y={0} stroke="#5E6168" strokeDasharray="3 3" />
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
          <Line
            type="monotone"
            dataKey="touch_watch_calibration_error"
            stroke="#7C9BB8"
            strokeWidth={1.5}
            dot={false}
            activeDot={{ r: 3, fill: "#9FB6CB" }}
            isAnimationActive={animate}
            animationBegin={animationBeginMs}
            animationDuration={animationDurationMs}
            animationEasing="ease-out"
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
