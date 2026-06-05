import {
  ARTIFACT_KIND_LABELS,
  relativeTimeFromNow,
} from "@/lib/artifacts";
import { getStorage } from "@/lib/artifact-storage";
import { LivePulse } from "./LivePulse";

// Server component — reads the artifact registry at render time
// and surfaces a live "freshly published" pill. Cached briefly via
// the parent page's revalidate setting.

export async function LatestPublishedTicker() {
  const store = await getStorage();
  const latest = (await store.list({ limit: 1 }))[0];
  if (!latest) {
    return (
      <LivePulse label="No publications yet — first brief lands tomorrow" />
    );
  }
  return (
    <LivePulse
      label={
        <>
          <span className="text-fg">
            {ARTIFACT_KIND_LABELS[latest.kind]}
          </span>{" "}
          published {relativeTimeFromNow(latest.generated_at_utc)}
        </>
      }
    />
  );
}
