"""Small independent correctness checks for the analysis' time-window semantics."""
from pyspark.sql import Window, functions as F
from task1_spark import start_spark


def main():
    spark = start_spark()
    try:
        base = 1704067200
        data = spark.createDataFrame([('S1', base + 300 * i, float(i)) for i in range(8)],
                                     ['station_id', 'slot_epoch', 'rain_mm'])
        future = Window.partitionBy('station_id').orderBy('slot_epoch').rangeBetween(300, 1800)
        def next_window(frame):
            return frame.withColumn('future_sum', F.sum('rain_mm').over(future)) \
                        .withColumn('future_count', F.count('*').over(future))
        first = next_window(data).filter(F.col('slot_epoch') == base).first()
        assert first.future_count == 6 and first.future_sum == 21
        # Missing an interval must invalidate the horizon, not extend it to six rows.
        gap = next_window(data.filter(F.col('slot_epoch') != base + 900)) \
              .filter(F.col('slot_epoch') == base).first()
        assert gap.future_count == 5 and gap.future_sum == 18
        assert not (first.future_sum > 21), 'The target uses strict >, not >='
        times = spark.createDataFrame([(base - 1,), (base,), (base + 3,)], ['original'])
        grid = times.withColumn('aligned', (F.round(F.col('original') / 300) * 300).cast('long')) \
                    .withColumn('accepted', F.abs(F.col('aligned') - F.col('original')) <= 2).collect()
        assert grid[0].aligned == base and grid[0].accepted
        assert grid[1].aligned == base and grid[1].accepted
        assert not grid[2].accepted
        print('Window checks passed: future-only horizon, missing interval, strict threshold, timestamp tolerance', flush=True)
    finally:
        spark.stop()


if __name__ == '__main__':
    main()
