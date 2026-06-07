export default function WordCloud({ terms }) {
  if (!terms || terms.length === 0) return null;

  const counts = terms.map(([, count]) => count);
  const minCount = Math.min(...counts);
  const maxCount = Math.max(...counts);
  const range = maxCount - minCount || 1;

  const getSize = (count) => {
    const normalized = (count - minCount) / range;
    return 0.9 + normalized * 1.8; // 0.9rem → 2.7rem
  };

  const getColor = (count) => {
    const normalized = (count - minCount) / range;
    const hue = 230 + normalized * 60; // indigo → violet
    const lightness = 55 - normalized * 20; // lighter → darker
    return `hsl(${hue}, 70%, ${lightness}%)`;
  };

  return (
    <div className="word-cloud-section">
      <h2 className="word-cloud-title">Aspect Term Word Cloud</h2>
      <div className="word-cloud-container">
        {terms.map(([term, count], idx) => (
          <span
            key={idx}
            className="word-cloud-term"
            style={{
              fontSize: `${getSize(count)}rem`,
              color: getColor(count),
              fontWeight: count === maxCount ? 700 : 400 + Math.round(((count - minCount) / range) * 3) * 100,
            }}
            title={`"${term}" — ${count} occurrence${count > 1 ? "s" : ""}`}
          >
            {term}
          </span>
        ))}
      </div>
    </div>
  );
}
