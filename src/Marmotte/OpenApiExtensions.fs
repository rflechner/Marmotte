namespace Marmotte

open System.Text.Json
open System.Threading
open Microsoft.AspNetCore.Mvc
open Microsoft.OpenApi.Models
open Microsoft.AspNetCore.OpenApi
open System.Reflection

module OpenApiExtensions =

    type IgnoreFromServicesSchemaTransformer() =
        interface IOpenApiSchemaTransformer with
            member _.TransformAsync(schema:OpenApiSchema, context:OpenApiSchemaTransformerContext, _:CancellationToken) =
                task {
                    let typ = context.JsonTypeInfo.Type
                    
                    if typ <> null && schema.Properties <> null then
                        typ.GetProperties(BindingFlags.Public ||| BindingFlags.Instance)
                        |> Array.filter (fun p -> p.GetCustomAttribute<FromServicesAttribute>() <> null)
                        |> Array.iter (fun p ->
                            let propName = JsonNamingPolicy.CamelCase.ConvertName(p.Name)
                            schema.Properties.Remove(propName) |> ignore
                        )

                    return schema
                }

