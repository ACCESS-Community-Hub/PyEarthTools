# Documentation For the Environmental Data Intelligence Toolkit

## Viewing Documentation Site

To view it locally run:

```bash
module load nodejs
npm install serve
npx serve site/
```

## Websites

https://bureau-sig.nci.org.au/pyearthtools/

## Updates

Documentation can be built using 
```shell
module use /scratch/ra02/modules
module load pyearthtools/documentation

mkdocs build ./
```